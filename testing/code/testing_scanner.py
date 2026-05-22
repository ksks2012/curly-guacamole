"""Smoke test for GCR1.1 — RepoScanner."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.code.scanner import RepoScanner
from rag.code.schema import RepoManifest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scan() -> RepoManifest:
    scanner  = RepoScanner()
    manifest = scanner.scan(".", repo_id="langchain-test")

    assert len(manifest) > 0, "manifest is empty"
    print(f"files={len(manifest)}  branch={manifest.branch!r}")

    langs: dict[str, int] = {}
    for f in manifest:
        if f.language:
            langs[f.language] = langs.get(f.language, 0) + 1
    for lang, cnt in sorted(langs.items(), key=lambda x: -x[1])[:8]:
        print(f"  {lang:<15} {cnt}")

    src = manifest.source_files()
    assert len(src) > 0, "no source files found"
    print(f"source_files (non-test/generated)={len(src)}")

    return manifest


def test_self_diff(manifest: RepoManifest) -> None:
    diff = RepoScanner.diff(manifest, manifest)
    assert diff.is_empty(), f"self-diff should be empty: {diff.summary()}"
    print(f"self_diff_empty={diff.is_empty()}")
    print(f"diff.summary={diff.summary()}")


def test_round_trip(manifest: RepoManifest) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "manifest.json"
        manifest.save(p)
        loaded = RepoManifest.load(p)

    assert len(loaded) == len(manifest), \
        f"round-trip count mismatch: {len(loaded)} vs {len(manifest)}"
    print(f"reload_count={len(loaded)}  match={len(loaded) == len(manifest)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    manifest = test_scan()
    test_self_diff(manifest)
    test_round_trip(manifest)
    print("PASS")


if __name__ == "__main__":
    main()
