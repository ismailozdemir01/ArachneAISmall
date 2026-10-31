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
