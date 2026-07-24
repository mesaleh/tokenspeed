"""CPU contracts for base-2 segmented-attention merge."""

import pytest
import torch

from tokenspeed_mla.mla_decode import merge_attention_outputs_base2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_merge_matches_concatenated_softmax(dtype: torch.dtype):
    torch.manual_seed(7)
    scores_a = torch.randn(1, 5, 8, 33, dtype=torch.float32)
    scores_b = torch.randn(1, 5, 8, 47, dtype=torch.float32)
    values_a = torch.randn(1, 5, 8, 33, 16, dtype=torch.float32)
    values_b = torch.randn(1, 5, 8, 47, 16, dtype=torch.float32)

    probs_a = torch.softmax(scores_a, dim=-1)
    probs_b = torch.softmax(scores_b, dim=-1)
    output_a = torch.einsum("bqhk,bqhkd->bqhd", probs_a, values_a).to(dtype)
    output_b = torch.einsum("bqhk,bqhkd->bqhd", probs_b, values_b).to(dtype)
    lse_a = torch.logsumexp(scores_a, dim=-1) / torch.log(torch.tensor(2.0))
    lse_b = torch.logsumexp(scores_b, dim=-1) / torch.log(torch.tensor(2.0))

    output, lse = merge_attention_outputs_base2(
        output_a, lse_a, output_b, lse_b
    )

    scores = torch.cat((scores_a, scores_b), dim=-1)
    values = torch.cat((values_a, values_b), dim=-2)
    reference = torch.einsum(
        "bqhk,bqhkd->bqhd", torch.softmax(scores, dim=-1), values
    )
    reference_lse = torch.logsumexp(scores, dim=-1) / torch.log(torch.tensor(2.0))
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(output.float(), reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, reference_lse, atol=2e-6, rtol=2e-6)


def test_merge_rejects_shape_and_dtype_drift():
    output = torch.empty(1, 1, 8, 16)
    lse = torch.empty(1, 1, 8)
    with pytest.raises(ValueError, match="output shapes"):
        merge_attention_outputs_base2(output, lse, output[..., :8], lse)
    with pytest.raises(ValueError, match="output dtypes"):
        merge_attention_outputs_base2(output, lse, output.bfloat16(), lse)
    with pytest.raises(ValueError, match="LSE shapes"):
        merge_attention_outputs_base2(output, lse[..., :4], output, lse)
    with pytest.raises(ValueError, match="must be FP32"):
        merge_attention_outputs_base2(
            output, lse.bfloat16(), output, lse.bfloat16()
        )
