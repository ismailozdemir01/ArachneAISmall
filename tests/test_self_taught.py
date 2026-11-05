from arachne_small.self_taught import (
    _is_approved_license,
    _is_living_person_category,
    _bootstrap_config,
)


def test_wikipedia_page_license_allowlist_accepts_cc_by_sa_40():
    result = _is_approved_license({
        "license": {
            "title": "Creative Commons Attribution-Share Alike 4.0",
            "url": "https://creativecommons.org/licenses/by-sa/4.0/",
        }
    })
    assert result == (
        "Creative Commons Attribution-Share Alike 4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
    )


def test_wikipedia_page_license_allowlist_accepts_cc_by_sa_30():
    result = _is_approved_license({
        "license": {
            "title": "Creative Commons Attribution-Share Alike 3.0",
            "url": "https://creativecommons.org/licenses/by-sa/3.0/",
        }
    })
    assert result is not None


def test_wikipedia_page_license_allowlist_rejects_unverified_or_incompatible_rights():
    assert _is_approved_license({"license": {"title": "Unknown", "url": "https://example.com/license"}}) is None
    assert _is_approved_license({"license": {"title": "CC BY-NC", "url": "https://creativecommons.org/licenses/by-nc/4.0/"}}) is None
    assert _is_approved_license({}) is None


def test_living_people_categories_are_excluded_in_both_languages():
    assert _is_living_person_category("en", [{"title": "Category:Living people"}])
    assert _is_living_person_category("tr", [{"title": "Kategori:Yaşayan insanlar"}])
    assert not _is_living_person_category("en", [{"title": "Category:Ancient historians"}])


def test_bootstrap_model_is_real_small_transformer_configuration(tmp_path):
    config = _bootstrap_config(1024, tmp_path / "run")
    assert config.model.vocab_size == 1024
    assert config.model.parameter_count() > 1_000_000
    assert config.model.parameter_count() < 20_000_000
    assert config.training.max_steps > 0
    assert config.training.output_dir == str(tmp_path / "run")
