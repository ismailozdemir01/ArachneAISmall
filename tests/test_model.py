import torch

from arachne_small.config import ModelConfig
from arachne_small.model import ArachneSmall


def small_config():
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        intermediate_size=64,
    )


def test_parameter_counter_matches_actual_parameters():
    config = small_config()
    model = ArachneSmall(config)
    assert config.parameter_count() == sum(parameter.numel() for parameter in model.parameters())


def test_forward_computes_next_token_loss():
    model = ArachneSmall(small_config())
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    result = model(input_ids, labels)
    assert result["logits"].shape == (2, 8, 64)
    assert result["loss"] is not None
    assert torch.isfinite(result["loss"])


def test_forward_rejects_out_of_vocabulary_tokens():
    model = ArachneSmall(small_config())
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 64]])
    try:
        model(tokens)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("Out-of-vocabulary token was accepted.")


def test_generate_accepts_top_k_zero_as_unfiltered_sampling():
    model = ArachneSmall(small_config())
    prompt = torch.tensor([[1, 2, 3]])
    result = model.generate(prompt, max_new_tokens=2, top_k=0)
    assert result.shape == (1, 5)


def test_gradient_checkpointing_supports_backward():
    config = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4,
        n_kv_heads=2, max_seq_len=16, intermediate_size=64,
        gradient_checkpointing=True,
    )
    model = ArachneSmall(config)
    model.train()
    tokens = torch.randint(0, 64, (2, 8))
    result = model(tokens, tokens)
    result["loss"].backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_generation_rejects_non_finite_temperature():
    model = ArachneSmall(small_config())
    try:
        model.generate(torch.tensor([[1, 2]]), temperature=float("nan"))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Non-finite temperature was accepted.")

def test_rmsnorm_preserves_low_precision_activation_dtype():
    from arachne_small.model import RMSNorm

    norm = RMSNorm(32, 1e-5)
    values = torch.randn(2, 4, 32, dtype=torch.bfloat16, requires_grad=True)
    output = norm(values)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()
