import json
import urllib.error

from arachne_small.global_research import GlobalWebResearch, _PageParser


def test_global_research_accepts_only_explicit_allowlisted_reuse_licenses():
    assert GlobalWebResearch._approved_license(
        ["https://creativecommons.org/licenses/by/4.0/"], "https://example.org/page"
    ) == ("4.0", "https://creativecommons.org/licenses/by/4.0/")
    assert GlobalWebResearch._approved_license(
        ["https://example.org/terms"], "https://example.org/page"
    ) is None
    assert GlobalWebResearch._approved_license([], "https://example.org/page") is None


def test_global_research_rejects_non_https_credentials_and_nonstandard_ports():
    valid = GlobalWebResearch._valid_url
    assert valid("https://example.org/research")
    assert valid("https://example.org:443/research")
    assert not valid("http://example.org/research")
    assert not valid("https://user:pass@example.org/research")
    assert not valid("https://example.org:8443/research")
    assert not valid("https://localhost/research")


def test_html_parser_extracts_page_metadata_and_excludes_script_content():
    parser = _PageParser()
    parser.feed("""
      <html><head><title>Open research</title>
      <link rel="license" href="https://creativecommons.org/licenses/by/4.0/">
      <link rel="canonical" href="/canonical">
      </head><body><nav>navigation noise</nav>
      <main><p>Useful research text.</p><script>secret script payload</script></main>
      <a href="/related">Related</a></body></html>
    """)
    assert "Open research" in " ".join(parser.title_parts)
    assert parser.license_urls == ["https://creativecommons.org/licenses/by/4.0/"]
    assert parser.canonical == "/canonical"
    assert parser.links == ["/related"]
    assert "Useful research text." in " ".join(parser.parts)
    assert "secret script payload" not in " ".join(parser.parts)


def test_global_research_acquires_multiple_domains_and_records_provenance(tmp_path, monkeypatch):
    researcher = GlobalWebResearch(max_pages=4)
    monkeypatch.setattr(
        researcher, "_search",
        lambda query: ["https://one.example/a", "https://two.example/b"],
    )

    def read_page(url):
        host = url.split("/")[2]
        return {
            "text": "Openly licensed research content",
            "provenance": {
                "source": "global-web",
                "title": host,
                "url": url,
                "domain": host,
                "license": "4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "training_eligible": True,
                "retrieved_at": 1,
                "content_sha256": host,
            },
        }

    monkeypatch.setattr(researcher, "_read_page", read_page)
    output = tmp_path / "global.jsonl"
    result = researcher.acquire(output, queries=["science", "engineering"])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["records"] == 2
    assert result["unique_domains"] == 2
    assert {row["provenance"]["domain"] for row in rows} == {"one.example", "two.example"}
    assert all(row["provenance"]["license_url"].startswith("https://creativecommons.org/") for row in rows)


def test_global_research_does_not_publish_empty_training_corpus(tmp_path, monkeypatch):
    researcher = GlobalWebResearch(max_pages=2)
    monkeypatch.setattr(researcher, "_search", lambda _query: ["https://example.org/page"])
    monkeypatch.setattr(researcher, "_read_page", lambda _url: None)
    output = tmp_path / "empty.jsonl"
    try:
        researcher.acquire(output, queries=["science"])
    except RuntimeError as exc:
        assert "no pages with explicit approved reuse licenses" in str(exc)
    else:
        raise AssertionError("No unlicensed or empty corpus may be published.")
    assert not output.exists()


def test_global_research_persists_cross_cycle_source_deduplication(tmp_path, monkeypatch):
    researcher = GlobalWebResearch(max_pages=2)
    monkeypatch.setattr(researcher, "_search", lambda _query: ["https://example.org/research"])

    def read_page(url):
        return {
            "text": "Licensed scientific research text",
            "provenance": {
                "source": "global-web",
                "title": "Research",
                "url": url,
                "domain": "example.org",
                "license": "4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "training_eligible": True,
                "retrieved_at": 10,
                "content_sha256": "fixed-content-hash",
            },
        }

    monkeypatch.setattr(researcher, "_read_page", read_page)
    database = tmp_path / "seen.sqlite3"
    first = researcher.acquire(
        tmp_path / "first" / "global.jsonl", queries=["science"], seen_database=database
    )
    assert first["records"] == 1
    second_path = tmp_path / "second" / "global.jsonl"
    try:
        researcher.acquire(second_path, queries=["science"], seen_database=database)
    except RuntimeError as exc:
        assert "no pages with explicit approved reuse licenses" in str(exc)
    else:
        raise AssertionError("Previously ingested sources must not be trained again.")
    assert not second_path.exists()


def test_robots_not_found_is_treated_as_no_published_policy(monkeypatch):
    researcher = GlobalWebResearch(max_pages=1)

    def missing_robots(_url, _accept):
        raise urllib.error.HTTPError(
            "https://example.org/robots.txt", 404, "Not Found", {}, None
        )

    monkeypatch.setattr(researcher, "_request", missing_robots)
    assert researcher._allowed_by_robots("https://example.org/article")
