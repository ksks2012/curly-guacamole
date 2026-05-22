"""
GCR2.2 — Symbol Evolution Builder.

Pure function that processes a chronologically ordered sequence of
``FileSnapshot`` objects for a single file and produces a
``SymbolEvolution`` record per symbol.

Algorithm
---------
For each snapshot (oldest first):
  1. Symbols appearing for the first time  → introduced_in = commit_hash
  2. Symbols already seen whose body hash changed → modified_in += commit_hash
  3. Symbols present in previous snapshot but absent now → candidate for deleted_in

After the final snapshot:
  - Symbols absent from the last snapshot get deleted_in set to the commit
    where they were last seen to disappear.
  - Symbols still present → deleted_in = "".

Requirements
------------
- Snapshots must be for the **same** (repo_id, file_path).
- They must be sorted **oldest first** (ascending commit date).
- ``symbol_hashes`` on each snapshot enables modified_in detection.
  If empty (non-Python or parser not used), modified_in will always be [].
- ``renamed_from`` is always [] in GCR2.2; deferred to GCR3.
"""

from __future__ import annotations

from rag.code.schema import FileSnapshot, SymbolEvolution, _evolution_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_symbol_evolutions(
    snapshots: list[FileSnapshot],
) -> list[SymbolEvolution]:
    """Build SymbolEvolution records from a file's snapshot history.

    Parameters
    ----------
    snapshots : FileSnapshot objects for a single file, sorted oldest first.

    Returns
    -------
    One SymbolEvolution per unique symbol name seen across all snapshots.
    Returns [] when *snapshots* is empty.
    """
    if not snapshots:
        return []

    repo_id   = snapshots[0].repo_id
    file_path = snapshots[0].file_path

    # state: symbol_name → {introduced_in, modified_in, last_hash}
    state: dict[str, dict] = {}
    # deleted candidates: symbol_name → most-recent commit where it disappeared
    deleted: dict[str, str] = {}
    prev_symbols: set[str] = set()

    for snap in snapshots:
        current = set(snap.symbols)

        for sym in current:
            if sym not in state:
                # First appearance
                state[sym] = {
                    "introduced_in": snap.commit_hash,
                    "modified_in":   [],
                    "last_hash":     snap.symbol_hashes.get(sym, ""),
                }
                # Clear any prior deletion record (symbol was re-introduced)
                deleted.pop(sym, None)
            else:
                prev_hash = state[sym]["last_hash"]
                curr_hash = snap.symbol_hashes.get(sym, "")
                if curr_hash and prev_hash and curr_hash != prev_hash:
                    state[sym]["modified_in"].append(snap.commit_hash)
                if curr_hash:
                    state[sym]["last_hash"] = curr_hash
                # Clear stale deletion record if symbol came back
                deleted.pop(sym, None)

        # Symbols that vanished in this snapshot
        for sym in prev_symbols - current:
            deleted[sym] = snap.commit_hash

        prev_symbols = current

    # Build result — final snapshot determines alive/deleted status
    final_symbols = set(snapshots[-1].symbols)
    result: list[SymbolEvolution] = []
    for sym_name, s in state.items():
        result.append(SymbolEvolution(
            evolution_id=_evolution_id(repo_id, file_path, sym_name),
            symbol_name=sym_name,
            repo_id=repo_id,
            file_path=file_path,
            introduced_in=s["introduced_in"],
            modified_in=s["modified_in"],
            deleted_in=deleted.get(sym_name, "") if sym_name not in final_symbols else "",
            renamed_from=[],
        ))

    return result
