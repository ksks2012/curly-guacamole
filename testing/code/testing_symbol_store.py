"""Smoke test for GCR1.3 — Symbol Registry (SymbolStore)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.code.ast_parser import PythonASTParser
from rag.code.schema import Symbol, SYMBOL_TYPES
from rag.code.symbol_store import SymbolStore


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_SAMPLE = '''\
"""Module docstring."""


class Animal:
    """Base animal class."""

    def __init__(self, name: str) -> None:
        self._name = name

    def speak(self) -> str:
        """Return the animal sound."""
        return ""

    def __repr__(self) -> str:
        return f"Animal({self._name!r})"


class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"


def make_animal(kind: str) -> Animal:
    """Factory function."""

    def _validate(k: str) -> None:
        if not k:
            raise ValueError("kind must not be empty")

    _validate(kind)
    return Animal(kind)
'''


def _build_store() -> tuple[SymbolStore, PythonASTParser]:
    parser = PythonASTParser()
    chunks = parser.parse(_SAMPLE, file_path="animals.py", repo_id="test")
    store  = SymbolStore.from_chunks(chunks)
    return store, parser


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_count(store: SymbolStore) -> None:
    assert len(store) > 0, "store should not be empty"
    print(f"store: {store.summary()}")

    expected_names = {
        "<module>", "Animal", "Dog",
        "Animal.__init__", "Animal.speak", "Animal.__repr__",
        "Dog.speak",
        "make_animal", "make_animal._validate",
    }
    found_names = {s.symbol_name for s in store}
    for n in expected_names:
        assert n in found_names, f"missing symbol: {n!r}  found={sorted(found_names)}"


def test_symbol_types(store: SymbolStore) -> None:
    for sym in store:
        assert sym.symbol_type in SYMBOL_TYPES, \
            f"invalid symbol_type {sym.symbol_type!r} for {sym.symbol_name}"

    classes = store.by_type("class")
    methods = store.by_type("method")
    mods    = store.by_type("module")

    assert len(mods)    == 1, f"expected 1 module, got {len(mods)}"
    assert len(classes) == 2, \
        f"expected 2 classes, got {len(classes)}: {[c.symbol_name for c in classes]}"
    assert any(s.symbol_name == "Animal.__init__" for s in methods), \
        "missing Animal.__init__"


def test_visibility(store: SymbolStore) -> None:
    init_sym = next(s for s in store if s.symbol_name == "Animal.__init__")
    assert init_sym.visibility == "dunder", \
        f"__init__ visibility: {init_sym.visibility!r}"

    repr_sym = next(s for s in store if s.symbol_name == "Animal.__repr__")
    assert repr_sym.visibility == "dunder", \
        f"__repr__ visibility: {repr_sym.visibility!r}"

    validate_sym = next(s for s in store if s.symbol_name == "make_animal._validate")
    assert validate_sym.visibility == "private", \
        f"_validate visibility: {validate_sym.visibility!r}"

    speak_sym = next(s for s in store if s.symbol_name == "Animal.speak")
    assert speak_sym.visibility == "public", \
        f"speak visibility: {speak_sym.visibility!r}"


def test_parent_resolution(store: SymbolStore) -> None:
    animal_sym = next(
        s for s in store if s.symbol_name == "Animal" and s.symbol_type == "class"
    )
    init_sym = next(s for s in store if s.symbol_name == "Animal.__init__")
    assert init_sym.parent_symbol == animal_sym.symbol_id, (
        f"__init__.parent_symbol: {init_sym.parent_symbol!r}  "
        f"expected: {animal_sym.symbol_id!r}"
    )

    make_sym = next(s for s in store if s.symbol_name == "make_animal")
    assert make_sym.parent_symbol == "", \
        f"make_animal.parent_symbol should be empty: {make_sym.parent_symbol!r}"


def test_children_of(store: SymbolStore) -> None:
    animal_sym = next(
        s for s in store if s.symbol_name == "Animal" and s.symbol_type == "class"
    )
    child_names = {c.symbol_name for c in store.children_of(animal_sym.symbol_id)}
    assert "Animal.__init__" in child_names, f"Animal children: {child_names}"
    assert "Animal.speak"    in child_names
    assert "Animal.__repr__" in child_names


def test_find_get(store: SymbolStore) -> None:
    results = store.find("speak")
    assert len(results) >= 2, f"find('speak') should return >=2, got {len(results)}"

    exact = store.find("Dog", exact=True)
    assert len(exact) == 1 and exact[0].symbol_type == "class"

    animal_sym = next(
        s for s in store if s.symbol_name == "Animal" and s.symbol_type == "class"
    )
    by_id = store.get(animal_sym.symbol_id)
    assert by_id is not None and by_id.symbol_name == "Animal"


def test_by_file(store: SymbolStore) -> None:
    file_syms = store.by_file("animals.py")
    assert len(file_syms) == len(store), "all symbols should be in animals.py"


def test_symbol_id_format(store: SymbolStore) -> None:
    for sym in store:
        assert sym.symbol_id.startswith("test::animals.py::"), \
            f"bad symbol_id: {sym.symbol_id}"


def test_round_trip(store: SymbolStore) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "symbols.json"
        store.save(p)
        loaded = SymbolStore.load(p)

    assert len(loaded) == len(store), \
        f"round-trip count mismatch: {len(loaded)} vs {len(store)}"
    for sym in store:
        reloaded = loaded.get(sym.symbol_id)
        assert reloaded == sym, f"round-trip mismatch for {sym.symbol_name}"


def test_merge(store: SymbolStore, parser: PythonASTParser) -> None:
    store2 = SymbolStore.from_chunks(
        parser.parse(_SAMPLE, file_path="copy.py", repo_id="test"),
        repo_id="test",
    )
    merged = SymbolStore(repo_id="test")
    merged.merge(store)
    merged.merge(store2)
    assert len(merged) == len(store) + len(store2), "merge count mismatch"


def test_real_file() -> None:
    real_path = Path(__file__).resolve().parent.parent / "rag" / "engine.py"
    if not real_path.exists():
        return
    repo_root   = real_path.parent.parent
    parser      = PythonASTParser()
    real_chunks = parser.parse_file(real_path, repo_root=repo_root, repo_id="langchain-test")
    real_store  = SymbolStore.from_chunks(real_chunks)
    print(f"engine.py  → {real_store.summary()}")
    assert len(real_store) > 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    store, parser = _build_store()

    test_basic_count(store)
    test_symbol_types(store)
    test_visibility(store)
    test_parent_resolution(store)
    test_children_of(store)
    test_find_get(store)
    test_by_file(store)
    test_symbol_id_format(store)
    test_round_trip(store)
    test_merge(store, parser)
    test_real_file()

    print(f"\nsample.py  → {len(store)} symbols")
    for t in sorted({s.symbol_type for s in store}):
        names = sorted(s.symbol_name for s in store.by_type(t))
        print(f"  {t:12s}: {names}")

    print("\nPASS")


if __name__ == "__main__":
    main()
