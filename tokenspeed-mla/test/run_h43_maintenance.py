#!/usr/bin/env python3
"""Single-shot CT13+CT14 H43 qualification or decision maintenance runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analyze_h43_codebook_ab import validate_decision_contract
from h43_codebook_ab_common import canonical_json_digest, load_contract

SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")


def alarm_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("H43 experiment budget expired")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qualification", "decision"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preparation-state", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--decision-contract", type=Path)
    return parser.parse_args()


def run(
    command: list[str] | tuple[str, ...],
    *,
    timeout: int,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        text=True,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except BaseException:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    return completed


def remote(
    host: str,
    command: list[str],
    *,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            *SSH,
            f"mesaleh@{host}",
            f"sudo -n bash -lc {shlex.quote(shlex.join(command))}",
        ],
        timeout=timeout,
        check=check,
    )


def remote_script(
    host: str,
    script: str,
    arguments: list[str],
    *,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    remote_command = shlex.join(["sudo", "-n", "bash", "-s", "--", *arguments])
    command = [*SSH, f"mesaleh@{host}", remote_command]
    return run(command, timeout=timeout, input_text=script, check=check)


def gpu_health_mismatches(
    row: dict[str, str], contract: dict[str, Any], aggregate_ecc: str
) -> dict[str, tuple[Any, Any]]:
    expected = {
        "name": contract["machine"]["gpu_name"],
        "pstate": contract["machine"]["pstate"],
        "clocks.sm": str(contract["machine"]["sm_clock_mhz"]),
        "clocks.max.sm": str(contract["machine"]["sm_clock_mhz"]),
        "clocks_event_reasons.hw_slowdown": contract["telemetry"][
            "inactive_event_value"
        ],
        "clocks_event_reasons.sw_thermal_slowdown": contract["telemetry"][
            "inactive_event_value"
        ],
        "ecc.errors.uncorrected.volatile.total": "0",
        "ecc.errors.uncorrected.aggregate.total": aggregate_ecc,
        "gpu_recovery_action": contract["machine"]["recovery_action"],
        "fabric.state": contract["machine"]["fabric_state"],
        "fabric.status": contract["machine"]["fabric_status"],
    }
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    try:
        power = float(row["power.draw.instant"])
        limit = float(row["power.limit"])
    except (KeyError, ValueError) as error:
        mismatches["power"] = (str(error), "finite draw and limit")
    else:
        if not 0 < power <= limit == contract["machine"]["power_limit_w"]:
            mismatches["power"] = (power, limit)
    return mismatches


def install_remote(host: str, source: Path, destination: str) -> None:
    payload = source.read_text(encoding="utf-8")
    temporary = f"{destination}.tmp"
    run(
        [*SSH, f"mesaleh@{host}", f"sudo -n tee {shlex.quote(temporary)} >/dev/null"],
        timeout=60,
        input_text=payload,
    )
    remote(host, ["chown", "root:root", temporary], timeout=30)
    remote(host, ["chmod", "0755", temporary], timeout=30)
    remote(host, ["mv", temporary, destination], timeout=30)


def write_remote_root_file(
    host: str, path: str, content: str, mode: str = "0600"
) -> None:
    temporary = f"{path}.tmp"
    run(
        [*SSH, f"mesaleh@{host}", f"sudo -n tee {shlex.quote(temporary)} >/dev/null"],
        timeout=60,
        input_text=content,
    )
    remote(host, ["chown", "root:root", temporary], timeout=30)
    remote(host, ["chmod", mode, temporary], timeout=30)
    remote(host, ["mv", temporary, path], timeout=30)


def shell_config(values: dict[str, Any]) -> str:
    return "".join(
        f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()
    )


def gate_record_command(
    work: str,
    results: str,
    name: str,
    source_digest: str,
    evidence: list[str],
    commands: list[str],
) -> list[str]:
    command = [
        "python3",
        f"{work}/record_h43_gate.py",
        "--contract",
        f"{work}/h43_codebook_ab_contract.json",
        "--name",
        name,
        "--source-manifest-digest",
        source_digest,
        "--evidence-root",
        results,
    ]
    for path in evidence:
        command += ["--evidence", path]
    for description in commands:
        command += ["--command", description]
    return command


class Campaign:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.contract = load_contract(args.contract.resolve())
        self.contract_digest = canonical_json_digest(self.contract)
        self.state = json.loads(args.preparation_state.read_text(encoding="utf-8"))
        self.candidate = self.state["candidate"]
        self.reference = self.state["reference"]
        self.service = self.contract["accepted_service"]
        self.rank0 = self.service["rank0"]
        self.rank1 = self.service["rank1"]
        self.host0 = self.rank0["ssh_host"]
        self.host1 = self.rank1["ssh_host"]
        self.campaign = args.campaign
        self.local_output = args.local_output.resolve()
        self.remote_root = f"/var/lib/h43/{self.campaign}"
        self.results = f"{self.remote_root}/results"
        self.prep_root = self.state["remote_root"]
        self.work = f"{self.prep_root}/candidate/work"
        self.reference_work = f"{self.prep_root}/reference/work"
        self.container_name = f"ct13-h43-{self.campaign}"
        self.container_id: str | None = None
        self.nonce = secrets.token_hex(32)
        self.started_epoch = int(time.time())
        self.restore_attempted = False
        self.validation_passed = False
        self.service_stopped = False
        self.aggregate_ecc = ""
        self.pre_service_telemetry: dict[str, list[dict[str, str]]] = {}
        self.restore_config_values: dict[str, Any] = {}
        self.contexts = [int(value) for value in self.contract["contexts"]]
        self.resource_context = max(self.contexts)
        self.gpu_index = self.contract["machine"]["gpu_index"]
        self.experiment_deadline = 0.0
        self.downtime_started = 0.0

    def validate_inputs(self) -> None:
        if not re.fullmatch(r"[a-z0-9-]{1,40}", self.campaign):
            raise ValueError(
                "campaign must contain only lowercase letters, digits, hyphens"
            )
        if self.local_output.exists():
            raise ValueError(f"local output already exists: {self.local_output}")
        if self.state.get("status") != "PREPARED":
            raise ValueError("preparation state is not PREPARED")
        state_copy = dict(self.state)
        observed_state_digest = state_copy.pop("state_digest", None)
        if observed_state_digest != canonical_json_digest(state_copy):
            raise ValueError("preparation state digest mismatch")
        for identity in (self.candidate, self.reference):
            for field in (
                "source_manifest_digest",
                "installed_mla_sha256",
                "cache_artifact_digest",
                "prebuild_cold_sha256",
                "prebuild_warm_sha256",
                "prebuild_no_kernel_sha256",
            ):
                value = identity.get(field, "")
                if not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError(
                        f"preparation {identity.get('role')}.{field} is invalid"
                    )
            expected_cache = (
                f"/var/lib/h43-codebook-cache/{identity['source_manifest_digest']}"
            )
            if identity.get("cache_root") != expected_cache:
                raise ValueError("preparation cache root is not source-manifest keyed")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity.get("image_id", "")):
                raise ValueError("preparation image ID is invalid")
        if self.reference.get("commit") != self.contract["source_parent"]:
            raise ValueError("reference commit differs from the frozen source parent")
        if not re.fullmatch(r"/var/lib/h43/[a-z0-9-]+/preparation", self.prep_root):
            raise ValueError("preparation remote root is unsafe")
        repository = run(
            [
                "git",
                "-C",
                str(self.args.contract.resolve().parent),
                "rev-parse",
                "--show-toplevel",
            ],
            timeout=30,
        ).stdout.strip()
        head = run(
            ["git", "-C", repository, "rev-parse", "HEAD"], timeout=30
        ).stdout.strip()
        dirty = run(
            ["git", "-C", repository, "status", "--porcelain", "--untracked-files=all"],
            timeout=30,
        ).stdout
        if head != self.candidate.get("commit") or dirty:
            raise ValueError(
                f"maintenance requires the clean prepared commit; repo={repository} head={head}"
            )
        if self.args.mode == "decision":
            if self.args.decision_contract is None:
                raise ValueError("decision mode requires --decision-contract")
            decision = json.loads(
                self.args.decision_contract.read_text(encoding="utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
            failures: list[str] = []
            validate_decision_contract(decision, self.contract, failures)
            if failures:
                raise ValueError(f"decision contract is invalid: {failures}")
            for field in (
                "source_manifest_digest",
                "installed_mla_sha256",
                "cache_artifact_digest",
            ):
                if decision[field] != self.candidate[field]:
                    raise ValueError(
                        f"decision contract {field} differs from preparation"
                    )
            self.decision = decision
        elif self.args.decision_contract is not None:
            raise ValueError("qualification mode refuses a decision contract")

    def maintenance_contract(self) -> tuple[dict[str, int], str]:
        if self.args.mode == "qualification":
            return (
                self.contract["maintenance_seconds"]["qualification"],
                "qualification",
            )
        count = self.decision["processes_per_context"]
        key = f"decision_n{count}"
        return self.contract["maintenance_seconds"][key], key

    def consume_decision_window(self) -> None:
        if self.args.mode != "decision":
            return
        marker_path = f"{self.prep_root}/DECISION_WINDOW_CONSUMED.json"
        marker = {
            "schema_version": 1,
            "experiment": self.contract["experiment"],
            "contract_digest": self.contract_digest,
            "decision_contract_digest": self.decision["decision_contract_digest"],
            "campaign": self.campaign,
            "reserved_at_epoch": int(time.time()),
        }
        payload = json.dumps(marker, allow_nan=False, sort_keys=True) + "\n"
        script = """import os,sys
path,payload=sys.argv[1:]
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
try:
    os.write(fd,payload.encode())
    os.fsync(fd)
finally:
    os.close(fd)
"""
        remote(
            self.host0,
            ["python3", "-c", script, marker_path, payload],
            timeout=30,
        )

    def setup(self) -> None:
        self.local_output.mkdir(parents=True)
        for host in (self.host0, self.host1):
            if remote(
                host, ["test", "!", "-e", self.remote_root], timeout=30, check=False
            ).returncode:
                raise RuntimeError(f"remote campaign root already exists on {host}")
        self.verify_preparation()
        for host in (self.host0, self.host1):
            remote(host, ["install", "-d", "-m", "0755", self.remote_root], timeout=60)
        remote(
            self.host0,
            ["install", "-d", "-m", "0755", self.results, f"{self.results}/gates"],
            timeout=60,
        )
        for host in (self.host0, self.host1):
            remote(
                host, ["install", "-d", "-m", "0755", "/usr/local/libexec"], timeout=60
            )
        test_dir = self.args.contract.resolve().parent
        install_remote(
            self.host0,
            test_dir / "restore_h43_rank0.sh",
            "/usr/local/libexec/restore_h43_rank0.sh",
        )
        install_remote(
            self.host1,
            test_dir / "restore_h43_rank1.sh",
            "/usr/local/libexec/restore_h43_rank1.sh",
        )
        for host in (self.host0, self.host1):
            install_remote(
                host,
                test_dir / "alert_h43_maintenance.sh",
                "/usr/local/libexec/alert_h43_maintenance.sh",
            )
            install_remote(
                host,
                test_dir / "terminal_h43_maintenance.sh",
                "/usr/local/libexec/terminal_h43_maintenance.sh",
            )
        self.restore_config_values = {
            "H43_CAMPAIGN": self.campaign,
            "H43_NONCE": self.nonce,
            "H43_RANK0_HOSTNAME": self.rank0["physical_hostname"],
            "H43_RANK1_HOSTNAME": self.rank1["physical_hostname"],
            "H43_RANK0_CONTAINER_NAME": self.rank0["container_name"],
            "H43_RANK1_CONTAINER_NAME": self.rank1["container_name"],
            "H43_RANK0_CONTAINER_ID": self.rank0["container_id"],
            "H43_RANK1_CONTAINER_ID": self.rank1["container_id"],
            "H43_RANK0_PROCESS_MARKER": self.rank0["process_marker"],
            "H43_RANK1_PROCESS_MARKER": self.rank1["process_marker"],
            "H43_RANK0_MARKER": self.rank0["node_rank_marker"],
            "H43_RANK1_MARKER": self.rank1["node_rank_marker"],
            "H43_MARKER_ADDRESS": self.rank1["marker_address"],
            "H43_MARKER_PORT": self.rank1["marker_port"],
            "H43_HEALTH_URL": self.service["health_url"],
            "H43_COMMAND_TIMEOUT": self.contract["timeouts_seconds"]["command"],
            "H43_GPU_INDEX": self.contract["machine"]["gpu_index"],
            "H43_GPU_IDLE_TIMEOUT": self.contract["timeouts_seconds"]["gpu_idle"],
            "H43_RANK1_READY_TIMEOUT": self.contract["timeouts_seconds"]["rank1_ready"],
            "H43_MARKER_READY_TIMEOUT": self.contract["timeouts_seconds"][
                "marker_ready"
            ],
            "H43_HTTP_HEALTH_TIMEOUT": self.contract["timeouts_seconds"]["http_health"],
        }
        config = shell_config(self.restore_config_values)
        for host in (self.host0, self.host1):
            write_remote_root_file(host, f"{self.remote_root}/restore.conf", config)
        self.verify_marker_path()

    def verify_preparation(self) -> None:
        remote(
            self.host0,
            ["curl", "-fsS", "--max-time", "10", self.service["health_url"]],
            timeout=20,
        )
        for identity, work in (
            (self.candidate, self.work),
            (self.reference, self.reference_work),
        ):
            manifest = f"{work}/H43_D1_SOURCE_MANIFEST.sha256"
            observed_manifest = remote(
                self.host0, ["sha256sum", manifest], timeout=60
            ).stdout.split()[0]
            if observed_manifest != identity["source_manifest_digest"]:
                raise RuntimeError(f"{identity['role']} source-manifest digest changed")
            image = remote(
                self.host0,
                [
                    "docker",
                    "image",
                    "inspect",
                    identity["image_id"],
                    "--format",
                    '{{.Id}}|{{index .Config.Labels "com.omniva.h43.source_commit"}}|{{index .Config.Labels "com.omniva.h43.source_role"}}',
                ],
                timeout=60,
            ).stdout.strip()
            if (
                image
                != f"{identity['image_id']}|{identity['commit']}|{identity['role']}"
            ):
                raise RuntimeError(
                    f"{identity['role']} image identity/labels changed: {image}"
                )
            verify = "sha256sum --check --strict /work/H43_D1_SOURCE_MANIFEST.sha256"
            remote(
                self.host0,
                [
                    "docker",
                    "run",
                    "--rm",
                    "--volume",
                    f"{work}:/work:ro",
                    identity["image_id"],
                    "bash",
                    "-lc",
                    verify,
                ],
                timeout=180,
            )
            cache_code = (
                "import json,sys;sys.path.insert(0,sys.argv[1]);"
                "from pathlib import Path;"
                "from h43_codebook_ab_common import compiled_artifact_manifest,load_contract;"
                "c=load_contract(Path(sys.argv[1])/ 'h43_codebook_ab_contract.json');"
                "print(json.dumps(compiled_artifact_manifest(Path(sys.argv[2]),c['cache'])))"
            )
            cache = json.loads(
                remote(
                    self.host0,
                    ["python3", "-c", cache_code, work, identity["cache_root"]],
                    timeout=180,
                ).stdout
            )
            if cache["digest"] != identity["cache_artifact_digest"]:
                raise RuntimeError(
                    f"{identity['role']} compiled-artifact cache changed"
                )
            role_root = f"{self.prep_root}/{identity['role']}"
            for field, filename in (
                ("prebuild_cold_sha256", "prebuild-cold.json"),
                ("prebuild_warm_sha256", "prebuild-warm.json"),
                ("prebuild_no_kernel_sha256", "prebuild-no-kernel.json"),
            ):
                observed = remote(
                    self.host0, ["sha256sum", f"{role_root}/{filename}"], timeout=60
                ).stdout.split()[0]
                if observed != identity[field]:
                    raise RuntimeError(f"{identity['role']} {filename} changed")
            no_kernel = json.loads(
                remote(
                    self.host0,
                    ["cat", f"{role_root}/prebuild-no-kernel.json"],
                    timeout=60,
                ).stdout
            )
            if (
                no_kernel.get("status") != "PASS"
                or no_kernel.get("kernel_launch_rows") != 0
            ):
                raise RuntimeError(
                    f"{identity['role']} source-only prebuild launch proof failed"
                )

    def verify_marker_path(self) -> None:
        marker_root = f"{self.remote_root}/marker-preflight"
        marker_path = f"{marker_root}/{self.nonce}/index.html"
        marker = f"{self.nonce}|marker-preflight"
        remote(
            self.host1,
            ["install", "-d", "-m", "0700", f"{marker_root}/{self.nonce}"],
            timeout=30,
        )
        write_remote_root_file(self.host1, marker_path, marker + "\n", "0600")
        unit = f"h43-marker-preflight-{self.campaign}"
        try:
            remote(
                self.host1,
                [
                    "systemd-run",
                    "--quiet",
                    "--collect",
                    f"--unit={unit}",
                    "--property=RuntimeMaxSec=60",
                    "/usr/bin/python3",
                    "-m",
                    "http.server",
                    str(self.rank1["marker_port"]),
                    "--bind",
                    self.rank1["marker_address"],
                    "--directory",
                    f"{marker_root}/{self.nonce}",
                ],
                timeout=30,
            )
            deadline = time.monotonic() + 20
            observed = ""
            url = f"http://{self.rank1['marker_address']}:{self.rank1['marker_port']}/"
            while time.monotonic() < deadline:
                result = remote(
                    self.host0,
                    ["curl", "-fsS", "--max-time", "2", url],
                    timeout=10,
                    check=False,
                )
                observed = result.stdout.strip()
                if observed == marker:
                    break
                time.sleep(1)
            if observed != marker:
                raise RuntimeError(
                    "CT13 cannot verify the nonce-protected CT14 marker path"
                )
            (self.local_output / "marker-path-preflight.txt").write_text(
                f"PASS {self.rank1['marker_address']}:{self.rank1['marker_port']}\n",
                encoding="utf-8",
            )
        finally:
            remote(
                self.host1,
                ["systemctl", "stop", f"{unit}.service"],
                timeout=30,
                check=False,
            )

    def snapshot_and_verify(self) -> None:
        remote(
            self.host0,
            ["curl", "-fsS", "--max-time", "10", self.service["health_url"]],
            timeout=20,
        )
        query = ",".join(self.contract["telemetry"]["query_fields"])
        sample = remote(
            self.host0,
            [
                "nvidia-smi",
                f"--id={self.contract['machine']['gpu_index']}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            timeout=60,
        ).stdout.strip()
        fields = [part.strip() for part in sample.split(",")]
        mapping = dict(
            zip(self.contract["telemetry"]["query_fields"], fields, strict=True)
        )
        self.aggregate_ecc = mapping["ecc.errors.uncorrected.aggregate.total"]
        if not self.aggregate_ecc.isdigit():
            raise RuntimeError("aggregate ECC baseline is invalid")
        (self.local_output / "pre-maintenance-gpu0.csv").write_text(
            sample + "\n", encoding="utf-8"
        )
        for host in (self.host0, self.host1):
            output = remote(
                host,
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                timeout=60,
            ).stdout
            rows: list[dict[str, str]] = []
            for line in output.splitlines():
                values = [part.strip() for part in line.split(",")]
                rows.append(
                    dict(
                        zip(
                            self.contract["telemetry"]["query_fields"],
                            values,
                            strict=True,
                        )
                    )
                )
            if not rows:
                raise RuntimeError(f"pre-maintenance telemetry is empty on {host}")
            for row in rows:
                aggregate = row.get("ecc.errors.uncorrected.aggregate.total", "")
                if not aggregate.isdigit():
                    raise RuntimeError(
                        f"pre-maintenance aggregate ECC is invalid on {host}: {row}"
                    )
                mismatches = gpu_health_mismatches(row, self.contract, aggregate)
                if mismatches:
                    raise RuntimeError(
                        f"pre-maintenance GPU health mismatch on {host}: {mismatches}"
                    )
            self.pre_service_telemetry[host] = rows
            (self.local_output / f"pre-maintenance-telemetry-{host}.csv").write_text(
                output, encoding="utf-8"
            )

        for rank, host in ((self.rank0, self.host0), (self.rank1, self.host1)):
            script = """set -euo pipefail
name=$1
expected_id=$2
expected_image=$3
output=$4
expected_process_marker=$6
expected_log_marker=$7
observed_id=$(docker inspect --format '{{.Id}}' "${name}")
observed_image=$(docker inspect --format '{{.Image}}' "${name}")
state=$(docker inspect --format '{{.State.Status}}' "${name}")
if [[ ${observed_id} != "${expected_id}" || ${observed_image} != "${expected_image}" || ${state} != running ]]; then
  echo "accepted rank identity/state mismatch" >&2
  exit 1
fi
docker inspect "${name}" >"${output}-inspect.json"
docker top "${name}" -eo pid,args >"${output}-top.txt"
started_at=$(docker inspect --format '{{.State.StartedAt}}' "${name}")
top=$(docker top "${name}" -eo pid,args)
logs=$(docker logs --since "${started_at}" "${name}" 2>&1)
if [[ ${top} != *"${expected_process_marker}"* || ${logs} != *"${expected_log_marker}"* ]]; then
  echo "accepted rank does not satisfy the frozen restore predicate" >&2
  exit 1
fi
nvidia-smi -q >"${output}-nvidia-smi-q.txt"
journalctl -k --since "@$5" --no-pager >"${output}-journal-baseline.txt"
"""
            remote_script(
                host,
                script,
                [
                    rank["container_name"],
                    rank["container_id"],
                    rank["image_id"],
                    f"{self.remote_root}/snapshot-{host}",
                    str(self.started_epoch),
                    rank["process_marker"],
                    rank["node_rank_marker"],
                ],
                timeout=120,
                check=True,
            )
            query = ",".join(self.contract["telemetry"]["query_fields"])
            telemetry = remote(
                host,
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                timeout=60,
            ).stdout
            fields = self.contract["telemetry"]["query_fields"]
            rows = [
                dict(
                    zip(
                        fields,
                        [part.strip() for part in line.split(",")],
                        strict=True,
                    )
                )
                for line in telemetry.splitlines()
                if line.strip()
            ]
            baseline_by_uuid = {
                row["uuid"]: row for row in self.pre_service_telemetry[host]
            }
            if {row["uuid"] for row in rows} != set(baseline_by_uuid):
                raise RuntimeError(f"pre-stop GPU identity set changed on {host}")
            for row in rows:
                baseline = baseline_by_uuid[row["uuid"]]
                mismatches = gpu_health_mismatches(
                    row,
                    self.contract,
                    baseline["ecc.errors.uncorrected.aggregate.total"],
                )
                if mismatches:
                    raise RuntimeError(
                        f"pre-stop GPU health mismatch on {host}: {mismatches}"
                    )
            (self.local_output / f"pre-stop-telemetry-{host}.csv").write_text(
                telemetry, encoding="utf-8"
            )

    def arm(self) -> None:
        maintenance, _ = self.maintenance_contract()
        try:
            for host, script_name in (
                (self.host0, "restore_h43_rank0.sh"),
                (self.host1, "restore_h43_rank1.sh"),
            ):
                command = [
                    "systemd-run",
                    "--quiet",
                    f"--unit=h43-restore-{self.campaign}",
                    f"--on-active={maintenance['failsafe']}s",
                    "--on-unit-active=60s",
                    "--timer-property=AccuracySec=1s",
                    f"/usr/local/libexec/{script_name}",
                    f"{self.remote_root}/restore.conf",
                ]
                remote(host, command, timeout=60)
                remote(
                    host,
                    [
                        "systemd-run",
                        "--quiet",
                        f"--unit=h43-alert-{self.campaign}",
                        f"--on-active={maintenance['alert']}s",
                        "--timer-property=AccuracySec=1s",
                        "/usr/local/libexec/alert_h43_maintenance.sh",
                        self.campaign,
                    ],
                    timeout=60,
                )
                remote(
                    host,
                    [
                        "systemd-run",
                        "--quiet",
                        f"--unit=h43-terminal-{self.campaign}",
                        f"--on-active={maintenance['terminal']}s",
                        "--timer-property=AccuracySec=1s",
                        "/usr/local/libexec/terminal_h43_maintenance.sh",
                        self.campaign,
                    ],
                    timeout=60,
                )
                remote(
                    host,
                    [
                        "systemctl",
                        "is-active",
                        f"h43-restore-{self.campaign}.timer",
                    ],
                    timeout=30,
                )
                remote(
                    host,
                    ["systemctl", "is-active", f"h43-alert-{self.campaign}.timer"],
                    timeout=30,
                )
                remote(
                    host,
                    ["systemctl", "is-active", f"h43-terminal-{self.campaign}.timer"],
                    timeout=30,
                )
        except BaseException:
            if self.service_stopped:
                raise
            for host in (self.host0, self.host1):
                for unit in (
                    f"h43-restore-{self.campaign}.timer",
                    f"h43-alert-{self.campaign}.timer",
                    f"h43-terminal-{self.campaign}.timer",
                ):
                    remote(host, ["systemctl", "stop", unit], timeout=30, check=False)
            raise

    def stop_accepted(self) -> None:
        self.service_stopped = True
        for rank, host in ((self.rank0, self.host0), (self.rank1, self.host1)):
            remote(
                host,
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=10s",
                    "60s",
                    "docker",
                    "stop",
                    "--time",
                    "30",
                    rank["container_id"],
                ],
                timeout=75,
            )
        idle_timeout = self.contract["timeouts_seconds"]["gpu_idle"]
        idle_script = """set -euo pipefail
deadline=$((SECONDS + $1))
stable=0
while (( SECONDS < deadline )); do
  rows=$(nvidia-smi --id=$2 --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -z ${rows} ]]; then
    stable=$((stable + 1))
    (( stable >= 3 )) && exit 0
  else
    stable=0
  fi
  sleep 1
done
exit 1
"""
        remote_script(
            self.host0,
            idle_script,
            [str(idle_timeout), str(self.gpu_index)],
            timeout=idle_timeout + 10,
        )

    def start_candidate(self) -> None:
        if (
            remote(
                self.host0,
                ["docker", "container", "inspect", self.container_name],
                timeout=30,
                check=False,
            ).returncode
            == 0
        ):
            raise RuntimeError("candidate experiment container name already exists")
        command = [
            "docker",
            "create",
            "--name",
            self.container_name,
            "--hostname",
            self.contract["machine"]["container_hostname"],
            "--gpus",
            f'"device={self.gpu_index}"',
            "--cap-add",
            "SYS_ADMIN",
            "--ipc",
            "host",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"CUTE_DSL_CACHE_DIR={self.candidate['cache_root']}",
            "--env",
            f"H43_AOT_MANIFEST={self.candidate['cache_root']}/h43-aot-manifest.json",
            "--env",
            f"H43_AOT_EXPECTED_ENTRIES={self.contract['cache']['dense_dispatch_keys'] + self.contract['cache']['tq_dispatch_keys']}",
            "--env",
            f"H43_SOURCE_MANIFEST_DIGEST={self.candidate['source_manifest_digest']}",
            "--env",
            f"H43_INSTALLED_MLA_SHA256={self.candidate['installed_mla_sha256']}",
            "--env",
            f"H43_CACHE_ROOT={self.candidate['cache_root']}",
            "--env",
            f"H43_AGGREGATE_ECC_BASELINE={self.aggregate_ecc}",
            "--env",
            f"H43_EXPECTED_CACHE_DIGEST={self.candidate['cache_artifact_digest']}",
            "--volume",
            f"{self.work}:/work:ro",
            "--volume",
            f"{self.candidate['cache_root']}:{self.candidate['cache_root']}:ro",
            "--volume",
            f"{self.results}:/results",
            self.candidate["image_id"],
            "sleep",
            "infinity",
        ]
        self.container_id = remote(self.host0, command, timeout=120).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", self.container_id):
            raise RuntimeError("docker did not return an exact candidate container ID")
        self.restore_config_values.update(
            {
                "H43_EXPERIMENT_CONTAINER_NAME": self.container_name,
                "H43_EXPERIMENT_CONTAINER_ID": self.container_id,
            }
        )
        config = shell_config(self.restore_config_values)
        for host in (self.host0, self.host1):
            write_remote_root_file(host, f"{self.remote_root}/restore.conf", config)
        started = remote(
            self.host0, ["docker", "start", self.container_id], timeout=120
        ).stdout.strip()
        if started != self.container_id:
            raise RuntimeError("docker did not start the exact candidate container")

    def exec_candidate(
        self, arguments: list[str], timeout: int, *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        if self.container_id is None:
            raise RuntimeError("candidate container has not started")
        return remote(
            self.host0,
            ["docker", "exec", self.container_id, *arguments],
            timeout=timeout,
            check=check,
        )

    def remaining_experiment_seconds(self) -> int:
        remaining = int(self.experiment_deadline - time.monotonic())
        if remaining < 1:
            raise TimeoutError("H43 experiment budget is exhausted")
        return remaining

    def write_gate(self, name: str, evidence: list[str], commands: list[str]) -> None:
        output = f"{self.results}/gates/{name}.json"
        command = gate_record_command(
            self.work,
            self.results,
            name,
            self.candidate["source_manifest_digest"],
            evidence,
            commands,
        )
        shell = f"{shlex.join(command)} > {shlex.quote(output)}"
        remote(self.host0, ["bash", "-lc", shell], timeout=120)

    def qualification(self) -> None:
        manifest_log = f"{self.results}/source-manifest.log"
        command = (
            f"sha256sum --check --strict /work/H43_D1_SOURCE_MANIFEST.sha256 "
            f"> /results/source-manifest.log"
        )
        self.exec_candidate(["bash", "-lc", command], timeout=120)
        self.write_gate("source_manifest", ["source-manifest.log"], [command])

        pdl_log = f"{self.results}/pdl-source-order.json"
        pdl = "python3 /work/check_h43_pdl_source.py"
        result = self.exec_candidate(pdl.split(), timeout=120)
        write_remote_root_file(self.host0, pdl_log, result.stdout, "0644")
        self.write_gate("pdl_source_order", ["pdl-source-order.json"], [pdl])

        remaining = self.remaining_experiment_seconds()
        self.exec_candidate(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                f"{remaining}s",
                "/work/run_h43_codebook_ab.sh",
                "qualification-smoke",
                "/results",
                "-",
            ],
            timeout=remaining + 20,
        )

        raw_timeout = self.contract["timeouts_seconds"]["raw_word"]
        raw_command = (
            "mkdir -p /results/raw-word-cache && "
            "CUTE_DSL_CACHE_DIR=/results/raw-word-cache "
            f"timeout --signal=TERM --kill-after=15s {raw_timeout}s "
            "python3 /work/probe_tq4_prmt_codebook.py"
        )
        raw = self.exec_candidate(
            ["bash", "-lc", raw_command], timeout=raw_timeout + 20
        )
        write_remote_root_file(
            self.host0, f"{self.results}/raw-word.log", raw.stdout + raw.stderr, "0644"
        )
        self.write_gate("raw_word_probe", ["raw-word.log"], [raw_command])

        self.run_ncu_resource_gate()
        self.run_sanitizers()
        self.cache_persistence()
        remaining = self.remaining_experiment_seconds()
        self.exec_candidate(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                f"{remaining}s",
                "/work/run_h43_codebook_ab.sh",
                "qualification-preflight",
                "/results",
                "-",
            ],
            timeout=remaining + 20,
        )

    def run_ncu_resource_gate(self) -> None:
        ncu = self.contract["ncu"]
        metrics = ",".join(ncu["required_metrics"])
        common = [
            "ncu",
            "--nvtx",
            "--nvtx-include",
            ncu["nvtx_range"],
            "--metrics",
            metrics,
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
            "--force-overwrite",
        ]
        probe = [
            "python3",
            "/work/probe_h43_codebook_graph.py",
            "--contract",
            "/work/h43_codebook_ab_contract.json",
            "--context",
            str(self.resource_context),
            "--sequence",
            "1",
        ]
        ncu_timeout = self.contract["timeouts_seconds"]["ncu_each"]
        reference_name = f"ct13-h43-reference-{self.campaign}"
        reference_command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{ncu_timeout}s",
            "docker",
            "run",
            "--rm",
            "--name",
            reference_name,
            "--gpus",
            f'"device={self.gpu_index}"',
            "--cap-add",
            "SYS_ADMIN",
            "--ipc",
            "host",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"CUTE_DSL_CACHE_DIR={self.reference['cache_root']}",
            "--env",
            f"H43_AOT_MANIFEST={self.reference['cache_root']}/h43-aot-manifest.json",
            "--env",
            f"H43_AOT_EXPECTED_ENTRIES={self.contract['cache']['dense_dispatch_keys'] + self.contract['cache']['tq_dispatch_keys']}",
            "--env",
            f"H43_SOURCE_MANIFEST_DIGEST={self.reference['source_manifest_digest']}",
            "--env",
            f"H43_INSTALLED_MLA_SHA256={self.reference['installed_mla_sha256']}",
            "--volume",
            f"{self.reference_work}:/work:ro",
            "--volume",
            f"{self.reference['cache_root']}:{self.reference['cache_root']}:ro",
            "--volume",
            f"{self.results}:/results",
            self.reference["image_id"],
            *common,
            "--log-file",
            "/results/reference-ncu.csv",
            "--export",
            "/results/reference-ncu",
            *probe,
        ]
        try:
            reference_result = remote(
                self.host0, reference_command, timeout=ncu_timeout + 30
            )
        finally:
            remote(
                self.host0,
                ["docker", "rm", "--force", reference_name],
                timeout=60,
                check=False,
            )
        write_remote_root_file(
            self.host0,
            f"{self.results}/reference-ncu-target.log",
            reference_result.stdout + reference_result.stderr,
            "0644",
        )
        candidate_result = self.exec_candidate(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                f"{ncu_timeout}s",
                *common,
                "--log-file",
                "/results/candidate-ncu.csv",
                "--export",
                "/results/candidate-ncu",
                *probe,
            ],
            timeout=ncu_timeout + 30,
        )
        write_remote_root_file(
            self.host0,
            f"{self.results}/candidate-ncu-target.log",
            candidate_result.stdout + candidate_result.stderr,
            "0644",
        )
        compare_command = [
            "python3",
            f"{self.work}/compare_h43_ncu.py",
            "--contract",
            f"{self.work}/h43_codebook_ab_contract.json",
            "--reference",
            f"{self.results}/reference-ncu.csv",
            "--candidate",
            f"{self.results}/candidate-ncu.csv",
        ]
        shell = f"{shlex.join(compare_command)} > {self.results}/ncu-comparison.json"
        remote(self.host0, ["bash", "-lc", shell], timeout=120)
        self.write_gate(
            "ncu_resource_comparison",
            [
                "reference-ncu.csv",
                "candidate-ncu.csv",
                "reference-ncu-target.log",
                "candidate-ncu-target.log",
                "ncu-comparison.json",
            ],
            [
                shlex.join(reference_command),
                "docker exec CANDIDATE " + shlex.join([*common, *probe]),
                shlex.join(compare_command),
            ],
        )

    def run_sanitizers(self) -> None:
        timeout = self.contract["timeouts_seconds"]["sanitizer_each"]
        for tool in ("memcheck", "initcheck"):
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                f"{timeout}s",
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                str(self.contract["sanitizers"]["error_exit_code"]),
                "--target-processes",
                "all",
                "python3",
                "/work/probe_h43_codebook_graph.py",
                "--contract",
                "/work/h43_codebook_ab_contract.json",
                "--context",
                str(self.resource_context),
                "--sequence",
                "1",
            ]
            result = self.exec_candidate(command, timeout=timeout + 30, check=False)
            log = f"sanitizer-{tool}.log"
            write_remote_root_file(
                self.host0,
                f"{self.results}/{log}",
                result.stdout + result.stderr,
                "0644",
            )
            if result.returncode:
                raise RuntimeError(
                    f"{tool} failed ({result.returncode}); preserved {log}"
                )
            self.write_gate(f"sanitizer_{tool}", [log], [shlex.join(command)])

        probe = [
            "python3",
            "/work/probe_h43_tq_racecheck.py",
            "--contract",
            "/work/h43_codebook_ab_contract.json",
            "--context",
            str(self.resource_context),
            "--sequence",
            "1",
        ]
        common = [
            "compute-sanitizer",
            "--tool",
            "racecheck",
            "--error-exitcode",
            str(self.contract["sanitizers"]["error_exit_code"]),
            "--target-processes",
            "all",
        ]
        reference_name = f"ct13-h43-race-reference-{self.campaign}"
        reference_command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{timeout}s",
            "docker",
            "run",
            "--rm",
            "--name",
            reference_name,
            "--gpus",
            f'"device={self.gpu_index}"',
            "--cap-add",
            "SYS_ADMIN",
            "--ipc",
            "host",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"CUTE_DSL_CACHE_DIR={self.reference['cache_root']}",
            "--env",
            f"H43_AOT_MANIFEST={self.reference['cache_root']}/h43-aot-manifest.json",
            "--env",
            f"H43_AOT_EXPECTED_ENTRIES={self.contract['cache']['dense_dispatch_keys'] + self.contract['cache']['tq_dispatch_keys']}",
            "--env",
            f"H43_SOURCE_MANIFEST_DIGEST={self.reference['source_manifest_digest']}",
            "--env",
            f"H43_INSTALLED_MLA_SHA256={self.reference['installed_mla_sha256']}",
            "--volume",
            f"{self.reference_work}:/work:ro",
            "--volume",
            f"{self.reference['cache_root']}:{self.reference['cache_root']}:ro",
            "--volume",
            f"{self.results}:/results",
            self.reference["image_id"],
            *common,
            "--log-file",
            "/results/sanitizer-racecheck-reference.log",
            *probe,
        ]
        try:
            reference_result = remote(
                self.host0,
                reference_command,
                timeout=timeout + 30,
                check=False,
            )
        finally:
            remote(
                self.host0,
                ["docker", "rm", "--force", reference_name],
                timeout=60,
                check=False,
            )
        reference_target = "sanitizer-racecheck-reference-target.log"
        write_remote_root_file(
            self.host0,
            f"{self.results}/{reference_target}",
            reference_result.stdout + reference_result.stderr,
            "0644",
        )
        if reference_result.returncode:
            raise RuntimeError(
                "reference racecheck failed "
                f"({reference_result.returncode}); preserved {reference_target}"
            )

        candidate_command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{timeout}s",
            *common,
            "--log-file",
            "/results/sanitizer-racecheck-candidate.log",
            *probe,
        ]
        candidate_result = self.exec_candidate(
            candidate_command,
            timeout=timeout + 30,
            check=False,
        )
        candidate_target = "sanitizer-racecheck-candidate-target.log"
        write_remote_root_file(
            self.host0,
            f"{self.results}/{candidate_target}",
            candidate_result.stdout + candidate_result.stderr,
            "0644",
        )
        if candidate_result.returncode:
            raise RuntimeError(
                "candidate racecheck failed "
                f"({candidate_result.returncode}); preserved {candidate_target}"
            )

        comparison = "sanitizer-racecheck-comparison.json"
        compare_command = [
            "python3",
            f"{self.work}/check_h43_racecheck.py",
            "--contract",
            f"{self.work}/h43_codebook_ab_contract.json",
            "--context",
            str(self.resource_context),
            "--reference-log",
            f"{self.results}/sanitizer-racecheck-reference.log",
            "--candidate-log",
            f"{self.results}/sanitizer-racecheck-candidate.log",
            "--reference-target",
            f"{self.results}/{reference_target}",
            "--candidate-target",
            f"{self.results}/{candidate_target}",
            "--reference-exit-code",
            str(reference_result.returncode),
            "--candidate-exit-code",
            str(candidate_result.returncode),
        ]
        shell = f"{shlex.join(compare_command)} > {self.results}/{comparison}"
        remote(self.host0, ["bash", "-lc", shell], timeout=120)
        self.write_gate(
            "sanitizer_racecheck",
            [
                "sanitizer-racecheck-reference.log",
                "sanitizer-racecheck-candidate.log",
                reference_target,
                candidate_target,
                comparison,
            ],
            [
                shlex.join(reference_command),
                "docker exec CANDIDATE " + shlex.join(candidate_command),
                shlex.join(compare_command),
            ],
        )

    def cache_persistence(self) -> None:
        if self.container_id is None:
            raise RuntimeError("candidate container is absent")
        before = remote(
            self.host0,
            [
                "docker",
                "exec",
                self.container_id,
                "python3",
                "-c",
                (
                    "import json,sys;sys.path.insert(0,'/work');"
                    "from h43_codebook_ab_common import *;"
                    "c=load_contract(Path('/work/h43_codebook_ab_contract.json'));"
                    f"print(json.dumps(compiled_artifact_manifest(Path('{self.candidate['cache_root']}'),c['cache'])))"
                ),
            ],
            timeout=120,
        ).stdout.strip()
        observed = remote(
            self.host0,
            ["docker", "inspect", "--format", "{{.Id}}", self.container_id],
            timeout=60,
        ).stdout.strip()
        remote(
            self.host0,
            ["docker", "stop", "--time", "10", self.container_id],
            timeout=60,
        )
        state = remote(
            self.host0,
            ["docker", "inspect", "--format", "{{.State.Status}}", self.container_id],
            timeout=60,
        ).stdout.strip()
        if state != "exited":
            raise RuntimeError("candidate container did not stop for cache persistence")
        restarted = remote(
            self.host0, ["docker", "start", self.container_id], timeout=60
        ).stdout.strip()
        if observed != self.container_id or restarted != self.container_id:
            raise RuntimeError(
                "cache persistence did not restart the exact candidate container"
            )
        after = remote(
            self.host0,
            [
                "docker",
                "exec",
                self.container_id,
                "python3",
                "-c",
                (
                    "import json,sys;sys.path.insert(0,'/work');"
                    "from h43_codebook_ab_common import *;"
                    "c=load_contract(Path('/work/h43_codebook_ab_contract.json'));"
                    f"print(json.dumps(compiled_artifact_manifest(Path('{self.candidate['cache_root']}'),c['cache'])))"
                ),
            ],
            timeout=120,
        ).stdout.strip()
        before_value = json.loads(before)
        after_value = json.loads(after)
        if (
            before_value["digest"] != after_value["digest"]
            or before_value["digest"] != self.candidate["cache_artifact_digest"]
        ):
            raise RuntimeError(
                "compiled-artifact cache changed across container restart"
            )
        evidence = {
            "status": "PASS",
            "container_id": self.container_id,
            "cache_before": before_value,
            "cache_after": after_value,
        }
        write_remote_root_file(
            self.host0,
            f"{self.results}/cache-persistence.json",
            json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
            "0644",
        )
        self.write_gate(
            "cache_persistence",
            ["cache-persistence.json"],
            [
                "docker stop/start exact candidate container and compare immutable cache manifests"
            ],
        )

    def decision_run(self) -> None:
        assert self.args.decision_contract is not None
        content = self.args.decision_contract.read_text(encoding="utf-8")
        write_remote_root_file(
            self.host0, f"{self.results}/decision-contract.json", content, "0644"
        )
        remaining = self.remaining_experiment_seconds()
        self.exec_candidate(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                f"{remaining}s",
                "/work/run_h43_codebook_ab.sh",
                "decision-collect",
                "/results",
                "/results/decision-contract.json",
            ],
            timeout=remaining + 20,
        )

    def stop_candidate(self) -> None:
        if self.container_id is None:
            return
        remote(
            self.host0,
            ["docker", "stop", "--time", "10", self.container_id],
            timeout=60,
            check=False,
        )

    def restore(self) -> None:
        self.restore_attempted = True
        self.stop_candidate()
        timeouts = self.contract["timeouts_seconds"]
        rank1_timeout = timeouts["rank1_ready"] + timeouts["marker_ready"] + 30
        rank0_timeout = (
            timeouts["gpu_idle"]
            + timeouts["rank1_ready"]
            + timeouts["http_health"]
            + 60
        )
        for host, script, timeout in (
            (self.host1, "/usr/local/libexec/restore_h43_rank1.sh", rank1_timeout),
            (self.host0, "/usr/local/libexec/restore_h43_rank0.sh", rank0_timeout),
        ):
            remote(host, [script, f"{self.remote_root}/restore.conf"], timeout=timeout)

    def validate_restoration(self) -> None:
        for rank, host in ((self.rank0, self.host0), (self.rank1, self.host1)):
            value = remote(
                host,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.Id}}|{{.Image}}|{{.State.Status}}|{{.RestartCount}}",
                    rank["container_name"],
                ],
                timeout=60,
            ).stdout.strip()
            expected_prefix = f"{rank['container_id']}|{rank['image_id']}|running|"
            if not value.startswith(expected_prefix):
                raise RuntimeError(
                    f"restored rank identity/state mismatch: {host}: {value}"
                )
            (self.local_output / f"restore-{host}.txt").write_text(
                value + "\n", encoding="utf-8"
            )
        health = remote(
            self.host0,
            ["curl", "-fsS", "--max-time", "10", self.service["health_url"]],
            timeout=20,
        )
        (self.local_output / "restored-health.txt").write_text(
            health.stdout, encoding="utf-8"
        )
        models = remote(
            self.host0,
            ["curl", "-fsS", "--max-time", "30", self.service["models_url"]],
            timeout=40,
        )
        model_value = json.loads(models.stdout)
        if self.service["model_name"] not in {
            item.get("id") for item in model_value.get("data", [])
        }:
            raise RuntimeError(
                "restored model endpoint does not expose the frozen model"
            )
        (self.local_output / "restored-models.json").write_text(
            json.dumps(model_value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload = json.dumps(
            {
                "model": self.service["model_name"],
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly: H43 restore healthy",
                    }
                ],
                "temperature": 0,
                "max_tokens": 64,
            },
            separators=(",", ":"),
        )
        completion = remote(
            self.host0,
            [
                "curl",
                "-fsS",
                "--max-time",
                str(self.contract["timeouts_seconds"]["completion"]),
                "-H",
                "Content-Type: application/json",
                "-d",
                payload,
                self.service["completion_url"],
            ],
            timeout=self.contract["timeouts_seconds"]["completion"] + 20,
        )
        completion_value = json.loads(completion.stdout)
        if not completion_value.get("choices"):
            raise RuntimeError("restored endpoint completion has no choices")
        (self.local_output / "restored-completion.json").write_text(
            json.dumps(completion_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for host in (self.host0, self.host1):
            scan = remote(
                host,
                ["journalctl", "-k", "--since", f"@{self.started_epoch}", "--no-pager"],
                timeout=120,
            ).stdout
            (self.local_output / f"window-journal-{host}.txt").write_text(
                scan, encoding="utf-8"
            )
            if re.search(r"\bS?Xid\b", scan, re.IGNORECASE):
                raise RuntimeError(f"new Xid/SXid detected on {host}")
            query = ",".join(self.contract["telemetry"]["query_fields"])
            telemetry = remote(
                host,
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                timeout=60,
            ).stdout
            (self.local_output / f"restored-telemetry-{host}.csv").write_text(
                telemetry, encoding="utf-8"
            )
            fields = self.contract["telemetry"]["query_fields"]
            observed_rows = [
                dict(
                    zip(fields, [part.strip() for part in line.split(",")], strict=True)
                )
                for line in telemetry.splitlines()
                if line.strip()
            ]
            baseline_by_uuid = {
                row["uuid"]: row for row in self.pre_service_telemetry[host]
            }
            if {row["uuid"] for row in observed_rows} != set(baseline_by_uuid):
                raise RuntimeError(f"GPU identity set changed on {host}")
            for row in observed_rows:
                baseline = baseline_by_uuid[row["uuid"]]
                mismatches = gpu_health_mismatches(
                    row,
                    self.contract,
                    baseline["ecc.errors.uncorrected.aggregate.total"],
                )
                if mismatches:
                    raise RuntimeError(
                        f"restored GPU health mismatch on {host}: {mismatches}"
                    )
        self.validation_passed = True

    def finalize_qualification(self) -> None:
        xid_evidence = []
        for host in (self.host0, self.host1):
            local_path = self.local_output / f"window-journal-{host}.txt"
            remote_path = f"{self.results}/window-journal-{host}.txt"
            write_remote_root_file(
                self.host0, remote_path, local_path.read_text(encoding="utf-8"), "0644"
            )
            xid_evidence.append(f"window-journal-{host}.txt")
        self.write_gate(
            "window_xid_scan",
            xid_evidence,
            [
                f"journalctl -k --since @{self.started_epoch} on CT13 and CT14; no Xid/SXid"
            ],
        )
        smoke_path = (
            f"{self.results}/qualification/smoke/context{self.contexts[0]}/result.json"
        )
        codebook = remote(
            self.host0,
            [
                "python3",
                "-c",
                "import json,sys;print(json.load(open(sys.argv[1]))['codebook_sha256'])",
                smoke_path,
            ],
            timeout=60,
        ).stdout.strip()
        aggregate_command = [
            "python3",
            f"{self.work}/build_h43_qualification_gates.py",
            "--contract",
            f"{self.work}/h43_codebook_ab_contract.json",
            "--evidence-root",
            self.results,
            "--source-manifest-digest",
            self.candidate["source_manifest_digest"],
            "--installed-mla-sha256",
            self.candidate["installed_mla_sha256"],
            "--cache-artifact-digest",
            self.candidate["cache_artifact_digest"],
            "--codebook-sha256",
            codebook,
            "--aggregate-ecc-baseline",
            self.aggregate_ecc,
        ]
        remote(
            self.host0,
            [
                "bash",
                "-lc",
                f"{shlex.join(aggregate_command)} > {self.results}/qualification-gates.json",
            ],
            timeout=120,
        )
        analyze = [
            "python3",
            f"{self.work}/analyze_h43_codebook_ab.py",
            "--stage",
            "pilot",
            "--root",
            self.results,
            "--contract",
            f"{self.work}/h43_codebook_ab_contract.json",
            "--qualification-gates",
            f"{self.results}/qualification-gates.json",
        ]
        remote(
            self.host0,
            [
                "bash",
                "-lc",
                f"{shlex.join(analyze)} > {self.results}/decision-contract.json",
            ],
            timeout=180,
        )
        status = remote(
            self.host0,
            [
                "python3",
                "-c",
                "import json,sys;print(json.load(open(sys.argv[1]))['status'])",
                f"{self.results}/decision-contract.json",
            ],
            timeout=60,
        ).stdout.strip()
        if status != "READY":
            raise RuntimeError(
                f"qualification produced {status}; no decision window is authorized"
            )

    def finalize_decision(self) -> None:
        validity = [
            "python3",
            f"{self.work}/analyze_h43_codebook_ab.py",
            "--stage",
            "validity",
            "--root",
            self.results,
            "--contract",
            f"{self.work}/h43_codebook_ab_contract.json",
            "--decision-contract",
            f"{self.results}/decision-contract.json",
        ]
        remote(
            self.host0,
            ["bash", "-lc", f"{shlex.join(validity)} > {self.results}/validity.json"],
            timeout=180,
        )
        status = remote(
            self.host0,
            [
                "python3",
                "-c",
                "import json,sys;print(json.load(open(sys.argv[1]))['status'])",
                f"{self.results}/validity.json",
            ],
            timeout=60,
        ).stdout.strip()
        if status != "VALID":
            raise RuntimeError(
                f"decision validity stage produced {status}; performance stage is forbidden"
            )
        performance = [
            "python3",
            f"{self.work}/analyze_h43_codebook_ab.py",
            "--stage",
            "performance",
            "--root",
            self.results,
            "--contract",
            f"{self.work}/h43_codebook_ab_contract.json",
            "--decision-contract",
            f"{self.results}/decision-contract.json",
            "--validity-result",
            f"{self.results}/validity.json",
        ]
        remote(
            self.host0,
            [
                "bash",
                "-lc",
                f"{shlex.join(performance)} > {self.results}/performance.json",
            ],
            timeout=300,
        )

    def disarm(self) -> None:
        if not self.validation_passed:
            raise RuntimeError(
                "refusing to disarm restore timers before full validation"
            )
        for host in (self.host0, self.host1):
            for unit in (
                f"h43-restore-{self.campaign}.timer",
                f"h43-alert-{self.campaign}.timer",
                f"h43-terminal-{self.campaign}.timer",
            ):
                remote(host, ["systemctl", "stop", unit], timeout=30, check=False)
            deadline = time.monotonic() + 60
            restore_service = f"h43-restore-{self.campaign}.service"
            while time.monotonic() < deadline:
                active = remote(
                    host,
                    ["systemctl", "is-active", restore_service],
                    timeout=15,
                    check=False,
                ).stdout.strip()
                if active not in {"active", "activating", "deactivating"}:
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"restore service did not quiesce on {host}")
            remote(
                host,
                ["systemctl", "stop", f"h43-marker-{self.campaign}.service"],
                timeout=30,
                check=False,
            )
            for service in (
                f"h43-restore-{self.campaign}.service",
                f"h43-alert-{self.campaign}.service",
            ):
                remote(
                    host,
                    ["systemctl", "reset-failed", service],
                    timeout=30,
                    check=False,
                )

    def collect(self) -> None:
        for host in (self.host0, self.host1):
            for suffix in (
                "inspect.json",
                "top.txt",
                "nvidia-smi-q.txt",
                "journal-baseline.txt",
            ):
                remote_path = f"{self.remote_root}/snapshot-{host}-{suffix}"
                value = remote(host, ["cat", remote_path], timeout=120, check=False)
                if value.returncode == 0:
                    (self.local_output / f"snapshot-{host}-{suffix}").write_text(
                        value.stdout, encoding="utf-8"
                    )
            unit_log = remote(
                host,
                [
                    "journalctl",
                    "--unit",
                    f"h43-restore-{self.campaign}.service",
                    "--since",
                    f"@{self.started_epoch}",
                    "--no-pager",
                ],
                timeout=120,
                check=False,
            )
            (self.local_output / f"restore-unit-journal-{host}.txt").write_text(
                unit_log.stdout + unit_log.stderr, encoding="utf-8"
            )
        remote_archive = f"{self.remote_root}/ct13-results.tar"
        remote(
            self.host0,
            ["tar", "-C", self.remote_root, "-cf", remote_archive, "results"],
            timeout=300,
        )
        remote(self.host0, ["chmod", "0644", remote_archive], timeout=30)
        run(
            [
                "scp",
                "-q",
                f"mesaleh@{self.host0}:{remote_archive}",
                str(self.local_output / "ct13-results.tar"),
            ],
            timeout=300,
        )
        remote(self.host0, ["rm", "-f", remote_archive], timeout=60, check=False)
        manifest = {}
        for path in sorted(self.local_output.rglob("*")):
            if path.is_file():
                manifest[path.relative_to(self.local_output).as_posix()] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
        (self.local_output / "LOCAL_EVIDENCE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def execute(self) -> None:
        self.validate_inputs()
        self.setup()
        self.snapshot_and_verify()
        self.consume_decision_window()
        self.arm()
        failure: BaseException | None = None
        maintenance, _ = self.maintenance_contract()
        try:
            previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
            self.experiment_deadline = time.monotonic() + maintenance["experiment"]
            signal.alarm(maintenance["experiment"])
            self.downtime_started = time.monotonic()
            self.stop_accepted()
            self.start_candidate()
            if self.args.mode == "qualification":
                self.qualification()
            else:
                self.decision_run()
        except BaseException as error:
            failure = error
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            if self.service_stopped:
                try:
                    signal.signal(signal.SIGALRM, alarm_handler)
                    terminal_deadline = self.downtime_started + maintenance["terminal"]
                    restore_remaining = int(terminal_deadline - time.monotonic())
                    if restore_remaining < 1:
                        raise TimeoutError("H43 terminal restore budget is exhausted")
                    signal.alarm(restore_remaining)
                    try:
                        self.restore()
                    finally:
                        signal.alarm(0)
                    validation_remaining = min(
                        self.contract["timeouts_seconds"]["final_validation"],
                        int(terminal_deadline - time.monotonic()),
                    )
                    if validation_remaining < 1:
                        raise TimeoutError(
                            "H43 terminal validation budget is exhausted"
                        )
                    signal.alarm(validation_remaining)
                    try:
                        self.validate_restoration()
                    finally:
                        signal.alarm(0)
                except BaseException as restore_error:
                    if failure is None:
                        failure = restore_error
                    else:
                        failure = ExceptionGroup(
                            "experiment and restoration both failed",
                            [failure, restore_error],
                        )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_handler)
        if failure is not None:
            if self.validation_passed:
                try:
                    self.disarm()
                except BaseException as disarm_error:
                    failure = ExceptionGroup(
                        "experiment and fail-safe disarm both failed",
                        [failure, disarm_error],
                    )
            raise failure
        try:
            if self.args.mode == "qualification":
                self.finalize_qualification()
            else:
                self.finalize_decision()
        finally:
            if self.validation_passed:
                self.disarm()
        self.collect()


def main() -> None:
    args = parse_args()
    campaign: Campaign | None = None
    try:
        campaign = Campaign(args)
        campaign.execute()
    except BaseException:
        if campaign is not None and campaign.local_output.exists():
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
