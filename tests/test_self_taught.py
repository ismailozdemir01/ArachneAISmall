from arachne_small.self_taught import (
    _is_approved_license,
    _is_living_person_category,
    _bootstrap_config,
    _CONTACT_OR_SECRET,
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



def test_acquisition_rejects_obvious_credentials_and_contact_details():
    assert _CONTACT_OR_SECRET.search("Contact user@example.com")
    assert _CONTACT_OR_SECRET.search("ghp_abcdefghijklmnopqrstuvwxyz123456")
    assert _CONTACT_OR_SECRET.search("AKIA1234567890ABCDEF")
    assert not _CONTACT_OR_SECRET.search("ArachneAI is a language model project.")



def test_acquisition_persists_page_deduplication_between_cycles(tmp_path, monkeypatch):
    import json
    import sqlite3
    import arachne_small.self_taught as self_taught

    batch = {"tr": 0, "en": 0}
    generation = {"value": 0}

    def fake_fetch(language):
        index = batch[language]
        batch[language] += 1
        page_id = generation["value"] * 20 + index + 1
        return {
            "text": f"<|language:{language}|>\nLicensed article {page_id} " + ("content " * 80),
            "provenance": {"language": language, "page_id": page_id},
        }

    monkeypatch.setattr(self_taught, "_fetch_page", fake_fetch)
    monkeypatch.setattr(self_taught.time, "sleep", lambda _seconds: None)
    seen_db = tmp_path / "seen.sqlite3"

    first = self_taught.acquire_wikipedia_corpus(
        tmp_path / "first.jsonl", articles_per_language=20,
        request_delay_seconds=1.0, max_attempts_per_language=20,
        seen_database=seen_db,
    )
    assert first["records"] == 40

    generation["value"] = 1
    batch.update({"tr": 0, "en": 0})
    second = self_taught.acquire_wikipedia_corpus(
        tmp_path / "second.jsonl", articles_per_language=20,
        request_delay_seconds=1.0, max_attempts_per_language=20,
        seen_database=seen_db,
    )
    assert second["records"] == 40

    with sqlite3.connect(seen_db) as database:
        count = database.execute("SELECT COUNT(*) FROM seen_pages").fetchone()[0]
    assert count == 80
    assert len((tmp_path / "first.jsonl").read_text(encoding="utf-8").splitlines()) == 40
    assert len((tmp_path / "second.jsonl").read_text(encoding="utf-8").splitlines()) == 40
