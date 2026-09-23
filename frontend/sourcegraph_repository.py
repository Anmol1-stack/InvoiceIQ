"""Live, evidence-only Sourcegraph search for the Repository Analysis tab.

This module deliberately has no dependency on the policy RAG services.  It is
kept separate so a repository question can only be answered from Sourcegraph
code-search evidence (or produce an explicit availability error).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class SourcegraphError(RuntimeError):
    """A safe error suitable for presenting in the dashboard."""


@dataclass(frozen=True)
class SourcegraphConfig:
    endpoint: str
    repository: str
    token: str

    @classmethod
    def from_environment(cls) -> "SourcegraphConfig":
        endpoint = os.getenv("SOURCEGRAPH_GRAPHQL_URL")
        if not endpoint:
            base_url = os.getenv("SOURCEGRAPH_URL", "https://sourcegraph.com").rstrip("/")
            endpoint = f"{base_url}/.api/graphql"
        repository = os.getenv("SOURCEGRAPH_REPOSITORY", os.getenv("SOURCEGRAPH_REPO", "")).strip()
        token = os.getenv("SOURCEGRAPH_ACCESS_TOKEN", os.getenv("SOURCEGRAPH_TOKEN", "")).strip()
        return cls(endpoint=endpoint, repository=repository, token=token)

    def validate(self) -> None:
        missing = []
        if not self.repository:
            missing.append("SOURCEGRAPH_REPOSITORY")
        if not self.token:
            missing.append("SOURCEGRAPH_ACCESS_TOKEN")
        if missing:
            raise SourcegraphError(
                "Sourcegraph is not configured. Set " + ", ".join(missing)
                + " (and optionally SOURCEGRAPH_URL or SOURCEGRAPH_GRAPHQL_URL), then try again."
            )


@dataclass(frozen=True)
class CodeLocation:
    path: str
    line: int | None
    snippet: str
    sourcegraph_url: str | None = None

    @property
    def reference(self) -> str:
        return f"{self.path}:{self.line}" if self.line is not None else self.path


_STOP_WORDS = {
    "what", "which", "where", "when", "does", "after", "with", "from", "that", "this",
    "would", "could", "should", "their", "there", "about", "have", "been", "into", "the",
    "and", "for", "are", "how", "files", "file", "function", "functions", "components",
    "code", "repository", "modified", "involved", "relevant", "happens", "user",
}


def _terms(question: str) -> list[str]:
    """Extract query terms from the user's question, rather than a canned answer."""
    backticked = re.findall(r"`([^`]+)`", question)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*", question.lower())
    values = backticked + [word for word in words if len(word) > 2 and word not in _STOP_WORDS]
    seen: set[str] = set()
    return [value for value in values if not (value.lower() in seen or seen.add(value.lower()))][:8]


def _repo_filter(repository: str) -> str:
    # Sourcegraph's repo filter is a regular expression. Escape the supplied
    # repository name so it cannot alter the scope of the search.
    return "repo:^" + re.escape(repository) + "$"


def build_searches(question: str, repository: str) -> list[dict[str, str]]:
    """Produce code-search queries for files, symbols, imports and callers.

    Every query is scoped to the configured repository. Search terms are based
    on the question; no InvoiceIQ dependency relationship is embedded here.
    """
    terms = _terms(question)
    if not terms:
        raise SourcegraphError("Enter a repository question containing a file, symbol, or code concept to search for.")

    scope = _repo_filter(repository)
    query_terms = " or ".join(re.escape(term) for term in terms)
    searches = [{
        "label": "Code and file matches",
        "query": f"{scope} type:file count:20 ({query_terms})",
    }]

    # Prefer an explicit code-like target over interrogative/action words.
    target = next(
        (
            term for term in terms
            if "." in term or "_" in term or term.lower() not in {"calls", "depends", "affected", "request", "flow", "service"}
        ),
        terms[0],
    )
    escaped_target = re.escape(target.rsplit(".", 1)[0])
    lowered = question.lower()
    if any(word in lowered for word in ("import", "depend", "caller", "calls", "affected")):
        searches.append({
            "label": "Imports, callers, and dependencies",
            "query": (
                f"{scope} type:file patternType:regexp count:20 "
                f"(import .*{escaped_target}|from .*{escaped_target}|{escaped_target}\\s*\\()"
            ),
        })
    if any(word in lowered for word in ("symbol", "function", "caller", "calls", "affected")):
        searches.append({
            "label": "Function and symbol definitions",
            "query": (
                f"{scope} type:file patternType:regexp count:20 "
                f"(def\\s+{escaped_target}|class\\s+{escaped_target}|{escaped_target}\\s*\\()"
            ),
        })
    return searches


class SourcegraphClient:
    """Minimal GraphQL client for Sourcegraph's stable code-search endpoint."""

    _QUERY = """
    query SearchCode($query: String!) {
      search(query: $query) {
        results {
          matchCount
          results {
            __typename
            ... on FileMatch {
              file { path url }
              lineMatches { preview lineNumber }
            }
          }
        }
      }
    }
    """

    def __init__(self, config: SourcegraphConfig | None = None):
        self.config = config or SourcegraphConfig.from_environment()

    def search(self, query: str) -> list[CodeLocation]:
        self.config.validate()
        payload = json.dumps({"query": self._QUERY, "variables": {"query": query}}).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"token {self.config.token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise SourcegraphError("Sourcegraph rejected the configured access token. Check SOURCEGRAPH_ACCESS_TOKEN.") from error
            raise SourcegraphError(f"Sourcegraph search failed (HTTP {error.code}).") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise SourcegraphError("Sourcegraph is unavailable. Check its URL, network access, and configuration.") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SourcegraphError("Sourcegraph returned an invalid search response.") from error

        errors = body.get("errors") or []
        if errors:
            message = errors[0].get("message", "unknown GraphQL error") if isinstance(errors[0], dict) else str(errors[0])
            raise SourcegraphError(f"Sourcegraph search could not run: {message}")

        raw_results = (((body.get("data") or {}).get("search") or {}).get("results") or {}).get("results") or []
        locations: list[CodeLocation] = []
        for item in raw_results:
            if item.get("__typename") != "FileMatch":
                continue
            file_data = item.get("file") or {}
            for match in item.get("lineMatches") or []:
                locations.append(CodeLocation(
                    path=file_data.get("path", "Unknown file"),
                    line=(match.get("lineNumber") + 1) if isinstance(match.get("lineNumber"), int) else None,
                    snippet=match.get("preview", ""),
                    sourcegraph_url=file_data.get("url"),
                ))
        return locations


def symbols_from_locations(locations: list[CodeLocation]) -> list[str]:
    """Extract source-defined/called symbols from returned Sourcegraph snippets."""
    symbols: list[str] = []
    for location in locations:
        symbols.extend(re.findall(r"\b(?:def|class)\s+([A-Za-z_]\w*)", location.snippet))
        symbols.extend(re.findall(r"\b([A-Za-z_]\w*)\s*\(", location.snippet))
    seen: set[str] = set()
    return [name for name in symbols if not (name in seen or seen.add(name))][:20]


def evidence_answer(question: str, locations: list[CodeLocation], symbols: list[str]) -> str:
    """Return a conservative conclusion derived only from retrieved evidence."""
    if not locations:
        return (
            f"Sourcegraph found no matching code locations in the configured repository for: {question!r}. "
            "Try a file name, endpoint, symbol, or service name."
        )
    references = ", ".join(location.reference for location in locations[:8])
    symbol_text = f" Observed symbols/calls include: {', '.join(symbols[:8])}." if symbols else ""
    return (
        f"Sourcegraph found {len(locations)} code-location match(es) for: {question!r}. "
        f"The strongest repository evidence is at {references}.{symbol_text} "
        "The snippets below are the live code-search evidence for this conclusion."
    )
