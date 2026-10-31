from arachne_small.config import load_config


def test_1_5b_configuration_is_in_target_parameter_class():
    config = load_config("configs/1_5b.yaml")
    count = config.model.parameter_count()
    assert count == 1_523_714_048


def test_development_configuration_loads():
    config = load_config("configs/dev.yaml")
    assert config.model.vocab_size == 512
    assert config.training.precision == "fp32"
