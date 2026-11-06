from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.robotparser
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

LOGGER = logging.getLogger("arachne_small.global_research")
USER_AGENT = "ArachneAISmall/0.2 (+https://github.com/ismailozdemir01/ArachneAISmall; independent research crawler)"
MAX_PAGE_BYTES = 3 * 1024 * 1024
MAX_TEXT_CHARS = 40000
MIN_TEXT_CHARS = 700
MAX_PAGES = 80
MAX_SEARCH_RESULTS = 25
ALLOWED_LICENSES = {
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by-sa/4.0/",
    "https://creativecommons.org/licenses/by/3.0/",
    "https://creativecommons.org/licenses/by-sa/3.0/",
    "https://creativecommons.org/licenses/by/2.0/",
    "https://creativecommons.org/licenses/by-sa/2.0/",
}
SECRET_PATTERN = re.compile(
    r"(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b)",
    re.IGNORECASE,
)
_TRACKING_HOSTS = {"localhost", "127.0.0.1", "::1"}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.skip_depth = 0
        self.license_urls: list[str] = []
        self.canonical: str | None = None
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}:
            self.skip_depth += 1
        if tag == "title":
            self.in_title = True
        rel = values.get("rel", "").casefold().split()
        if "license" in rel and values.get("href"):
            self.license_urls.append(values["href"])
        if "canonical" in rel and values.get("href"):
            self.canonical = values["href"]
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if tag == "meta":
            key = (values.get("name") or values.get("property") or "").casefold()
            if key in {"license", "dc.rights", "dcterms.rights", "copyright"}:
                candidate = values.get("content", "")
                if candidate.startswith(("https://", "http://")):
                    self.license_urls.append(candidate)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


class GlobalWebResearch:
    """Bounded, robots-aware web discovery with explicit reuse-rights verification."""

    def __init__(
        self,
        *,
        timeout: float = 18.0,
        max_pages: int = MAX_PAGES,
        per_host_delay: float = 2.0,
        search_endpoint: str | None = None,
    ) -> None:
        if not 1 <= max_pages <= 500:
            raise ValueError("max_pages must be between 1 and 500.")
        if timeout <= 0 or per_host_delay < 1:
            raise ValueError("timeout must be positive and per_host_delay at least one second.")
        self.timeout = timeout
        self.max_pages = max_pages
        self.per_host_delay = per_host_delay
        configured_endpoints = os.environ.get("ARACHNE_SEARCH_ENDPOINTS", "").strip()
        if search_endpoint:
            self.search_endpoints = (search_endpoint,)
        elif configured_endpoints:
            self.search_endpoints = tuple(item.strip() for item in configured_endpoints.split("||") if item.strip())
        else:
            self.search_endpoints = (
                "https://html.duckduckgo.com/html/",
                "https://www.mojeek.com/search",
            )
        if not self.search_endpoints or any(not self._valid_url(item) for item in self.search_endpoints):
            raise ValueError("Search endpoints must be HTTPS URLs on valid public hosts.")
        self._host_lock = threading.Lock()
        self._last_host_request: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._robots_lock = threading.Lock()

    @staticmethod
    def _valid_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.hostname.casefold() not in _TRACKING_HOSTS
                and parsed.username is None
                and parsed.password is None
                and (parsed.port is None or parsed.port == 443)
            )
        except ValueError:
            return False

    def _request(self, url: str, accept: str = "text/html,application/xhtml+xml") -> tuple[bytes, str, str]:
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
        with self._host_lock:
            elapsed = time.monotonic() - self._last_host_request.get(host, 0.0)
            if elapsed < self.per_host_delay:
                time.sleep(self.per_host_delay - elapsed)
            self._last_host_request[host] = time.monotonic()
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": accept, "Accept-Language": "en,tr;q=0.9,*;q=0.5"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            final_url = response.geturl()
            final = urlparse(final_url)
            if final.scheme != "https" or final.hostname != parsed.hostname:
                raise ValueError("Cross-origin or non-HTTPS redirect rejected.")
            content_type = response.headers.get_content_type().casefold()
            payload = response.read(MAX_PAGE_BYTES + 1)
        if len(payload) > MAX_PAGE_BYTES:
            raise ValueError("Response exceeded the page size limit.")
        return payload, final_url, content_type

    def _allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"https://{parsed.netloc}"
        with self._robots_lock:
            parser = self._robots.get(origin)
            if parser is None:
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(origin + "/robots.txt")
                try:
                    payload, _, _ = self._request(origin + "/robots.txt", "text/plain")
                    parser.parse(payload.decode("utf-8", errors="replace").splitlines())
                except urllib.error.HTTPError as exc:
                    # RFC 9309 treats 4xx robots responses as unavailable; 401/403 remain denied.
                    if exc.code == 404:
                        parser.parse([])
                    else:
                        return False
                except (OSError, ValueError, urllib.error.URLError):
                    # Network/server failures fail closed; do not crawl without a readable policy.
                    return False
                self._robots[origin] = parser
        return parser.can_fetch(USER_AGENT, url)

    def _search(self, query: str) -> list[str]:
        candidates: list[str] = []
        provider_errors: list[str] = []
        for endpoint in self.search_endpoints:
            url = endpoint + ("&" if "?" in endpoint else "?") + urllib.parse.urlencode({"q": query})
            try:
                payload, _, content_type = self._request(url)
                if "html" not in content_type:
                    raise ValueError("Search provider did not return HTML.")
                parser = _PageParser()
                parser.feed(payload.decode("utf-8", errors="replace"))
                provider_host = (urlparse(endpoint).hostname or "").casefold()
                for href in parser.links:
                    absolute = urljoin(url, html.unescape(href))
                    parsed = urlparse(absolute)
                    if parsed.hostname and (
                        parsed.hostname.casefold() == provider_host
                        or parsed.hostname.casefold().endswith(".duckduckgo.com")
                    ):
                        query_values = urllib.parse.parse_qs(parsed.query)
                        absolute = query_values.get("uddg", [absolute])[0]
                    absolute = urllib.parse.urldefrag(absolute).url
                    if self._valid_url(absolute) and absolute not in candidates:
                        candidates.append(absolute)
                    if len(candidates) >= MAX_SEARCH_RESULTS:
                        return candidates
            except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
                provider_errors.append(f"{endpoint}: {exc}")
                LOGGER.warning("Search provider %s failed: %s", endpoint, exc)
        if not candidates and provider_errors:
            raise RuntimeError("All configured global search providers failed: " + "; ".join(provider_errors))
        return candidates

    @staticmethod
    def _approved_license(license_urls: list[str], page_url: str) -> tuple[str, str] | None:
        for raw in license_urls:
            candidate = urljoin(page_url, html.unescape(raw)).strip()
            normalized = candidate.split("#", 1)[0].split("?", 1)[0].rstrip("/") + "/"
            for approved in ALLOWED_LICENSES:
                if normalized.casefold() == approved.casefold():
                    return approved.rsplit("/", 2)[-2].upper(), approved
        return None

    def _read_page(self, url: str) -> dict[str, Any] | None:
        if not self._valid_url(url) or not self._allowed_by_robots(url):
            return None
        payload, final_url, content_type = self._request(url)
        if content_type not in {"text/html", "application/xhtml+xml"}:
            return None
        parser = _PageParser()
        parser.feed(payload.decode("utf-8", errors="replace"))
        title = " ".join(" ".join(parser.title_parts).split())[:300]
        text = " ".join(" ".join(parser.parts).split())
        if len(text) < MIN_TEXT_CHARS or len(text) > MAX_TEXT_CHARS or SECRET_PATTERN.search(text):
            return None
        license_data = self._approved_license(parser.license_urls, final_url)
        canonical = urljoin(final_url, parser.canonical) if parser.canonical else final_url
        if not self._valid_url(canonical):
            canonical = final_url
        host = (urlparse(canonical).hostname or "").casefold()
        if not title or len(text.split()) < 100:
            return None
        digest = hashlib.sha256((canonical + "\n" + text).encode("utf-8")).hexdigest()
        return {
            "text": text,
            "provenance": {
                "source": "global-web",
                "title": title,
                "url": canonical,
                "domain": host,
                "license": license_data[0] if license_data else None,
                "license_url": license_data[1] if license_data else None,
                "training_eligible": license_data is not None,
                "retrieved_at": int(time.time()),
                "content_sha256": digest,
                "quality_signals": {
                    "word_count": len(text.split()),
                    "title_present": True,
                    "explicit_reuse_license": license_data is not None,
                    "robots_allowed": True,
                },
            },
        }

    def acquire(self, output_jsonl: str | Path, *, queries: list[str] | None = None, seen_database: str | Path | None = None) -> dict[str, Any]:
        target = Path(output_jsonl).expanduser()
        if target.is_symlink() or target.exists():
            raise FileExistsError(f"Research output already exists or is a symbolic link: {target}")
        configured = os.environ.get("ARACHNE_RESEARCH_QUERIES", "").strip()
        if queries is None:
            queries = [item.strip() for item in configured.split("||") if item.strip()] if configured else [
                "open scientific research mathematics physics biology computer science",
                "technical documentation open source engineering standards",
                "history linguistics geography archaeology open knowledge",
                "climate earth science astronomy public research datasets",
                "Turkish bilim araştırma mühendislik açık erişim",
                "multilingual educational resources open license",
            ]
        queries = [query.strip() for query in queries if isinstance(query, str) and query.strip()][:20]
        if not queries:
            raise ValueError("At least one non-empty global research query is required.")
        target.parent.mkdir(parents=True, exist_ok=True)
        index_target = target.with_name("research-index.jsonl")
        if index_target.is_symlink() or index_target.exists():
            raise FileExistsError(f"Research index already exists or is a symbolic link: {index_target}")
        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()
        domains: set[str] = set()
        accepted = 0
        scanned = 0
        research_pages = 0
        unlicensed_pages = 0
        searched = 0
        errors = 0
        database = None
        if seen_database is not None:
            database_path = Path(seen_database).expanduser()
            if database_path.is_symlink():
                raise ValueError("Global research deduplication database cannot be a symbolic link.")
            database_path.parent.mkdir(parents=True, exist_ok=True)
            database = sqlite3.connect(database_path, timeout=30)
            database.execute("PRAGMA journal_mode=WAL")
            database.execute(
                "CREATE TABLE IF NOT EXISTS seen_sources (content_sha256 TEXT PRIMARY KEY, url TEXT NOT NULL, first_seen INTEGER NOT NULL)"
            )
            database.execute("BEGIN IMMEDIATE")
        try:
            with target.open("x", encoding="utf-8", newline="\n") as output, index_target.open("x", encoding="utf-8", newline="\n") as index_output:
                for query in queries:
                    try:
                        urls = self._search(query)
                        searched += len(urls)
                    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
                        errors += 1
                        LOGGER.warning("Global search provider failed for query %r: %s", query, exc)
                        continue
                    for url in urls:
                        if accepted >= self.max_pages or scanned >= self.max_pages * 5:
                            break
                        scanned += 1
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        try:
                            record = self._read_page(url)
                        except (OSError, ValueError, urllib.error.URLError, UnicodeError) as exc:
                            errors += 1
                            LOGGER.debug("Rejected research candidate %s: %s", url, exc)
                            continue
                        if record is None:
                            continue
                        provenance = record["provenance"]
                        research_pages += 1
                        index_record = {
                            "title": provenance["title"],
                            "url": provenance["url"],
                            "domain": provenance["domain"],
                            "retrieved_at": provenance["retrieved_at"],
                            "content_sha256": provenance["content_sha256"],
                            "license": provenance["license"],
                            "license_url": provenance["license_url"],
                            "training_eligible": provenance["training_eligible"],
                            "snippet": record["text"][:400],
                        }
                        index_output.write(json.dumps(index_record, ensure_ascii=False, allow_nan=False) + "\n")
                        index_output.flush()
                        if not provenance["training_eligible"]:
                            unlicensed_pages += 1
                            continue
                        digest = provenance["content_sha256"]
                        canonical_url = provenance["url"]
                        if digest in seen_hashes:
                            continue
                        if database is not None and database.execute(
                            "SELECT 1 FROM seen_sources WHERE content_sha256=?",
                            (digest,),
                        ).fetchone() is not None:
                            continue
                        seen_hashes.add(digest)
                        domains.add(provenance["domain"])
                        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        output.flush()
                        if database is not None:
                            database.execute(
                                "INSERT INTO seen_sources(content_sha256,url,first_seen) VALUES(?,?,?)",
                                (digest, canonical_url, int(time.time())),
                            )
                        accepted += 1
                    if accepted >= self.max_pages:
                        break
            if accepted == 0:
                raise RuntimeError(
                    "Global research found no pages with explicit approved reuse licenses. "
                    "No training data was produced and the active model was not changed."
                )
            if database is not None:
                database.commit()
            return {
                "records": accepted,
                "research_pages": research_pages,
                "unlicensed_pages": unlicensed_pages,
                "research_index": str(index_target),
                "queries": len(queries),
                "search_results": searched,
                "unique_domains": len(domains),
                "domains": sorted(domains),
                "errors": errors,
                "source": "global-web-search-and-robots-aware-crawler",
                "license_policy": sorted(ALLOWED_LICENSES),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "output": str(target),
            }
        except Exception:
            if database is not None:
                database.rollback()
            target.unlink(missing_ok=True)
            raise
        finally:
            if database is not None:
                database.close()
