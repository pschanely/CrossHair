import pdb

from crosshair import type_repo


def test_skips_pdb_internal_types() -> None:
    script_target_type = getattr(pdb, "_ScriptTarget", None)
    if script_target_type is None:
        return
    original_map = type_repo._MAP
    type_repo.rebuild_subclass_map()
    try:
        subclass_map = type_repo.get_subclass_map()
        assert script_target_type not in subclass_map[str]
    finally:
        type_repo._MAP = original_map
