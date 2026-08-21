import pytest
from opencrl.groups import resolve_group, validate_index


@pytest.fixture
def index():
    """A mixed index: two modules, several levels, some tasks without a project."""
    return [
        {"name": "cybergym/libxml2-lvl0", "module": "cybergym", "project": "libxml2", "level": 0},
        {"name": "cybergym/libxml2-lvl1", "module": "cybergym", "project": "libxml2", "level": 1},
        {"name": "cybergym/openssl-lvl1", "module": "cybergym", "project": "openssl", "level": 1},
        {"name": "cybergym/unclaimed", "module": "cybergym", "project": None, "level": 2},
        {"name": "other/standalone", "module": "other", "project": "zlib", "level": 1},
    ]


# --- level filter -----------------------------------------------------------

def test_level_filter_selects_all_tasks_at_level(index):
    assert resolve_group("cybergym/level1", index) == [
        "cybergym/libxml2-lvl1",
        "cybergym/openssl-lvl1",
    ]


def test_level_filter_excludes_other_modules(index):
    names = resolve_group("cybergym/level1", index)
    assert "other/standalone" not in names


def test_level0_is_a_valid_level(index):
    assert resolve_group("cybergym/level0", index) == ["cybergym/libxml2-lvl0"]


# --- project filter ---------------------------------------------------------

def test_project_filter_selects_all_project_tasks(index):
    assert resolve_group("cybergym/project=libxml2", index) == [
        "cybergym/libxml2-lvl0",
        "cybergym/libxml2-lvl1",
    ]


def test_project_filter_ignores_projectless_tasks(index):
    names = resolve_group("cybergym/project=openssl", index)
    assert names == ["cybergym/openssl-lvl1"]
    assert "cybergym/unclaimed" not in names


def test_project_filter_scopes_to_module(index):
    # zlib lives only under `other`, not `cybergym`.
    with pytest.raises(ValueError, match="project"):
        resolve_group("cybergym/project=zlib", index)


# --- explicit task lists ----------------------------------------------------

def test_explicit_list_resolves_names(index):
    assert resolve_group(["cybergym/unclaimed", "other/standalone"], index) == [
        "cybergym/unclaimed",
        "other/standalone",
    ]


def test_explicit_list_preserves_input_order(index):
    assert resolve_group(["other/standalone", "cybergym/libxml2-lvl0"], index) == [
        "other/standalone",
        "cybergym/libxml2-lvl0",
    ]


def test_bare_string_resolves_as_single_task_name(index):
    assert resolve_group("other/standalone", index) == ["other/standalone"]


# --- mixed cases ------------------------------------------------------------

def test_level_filter_over_shuffled_index_is_sorted(index):
    shuffled = list(reversed(index))
    assert resolve_group("cybergym/level1", shuffled) == [
        "cybergym/libxml2-lvl1",
        "cybergym/openssl-lvl1",
    ]


def test_default_keys_missing_project_and_level(index):
    # Index entries may omit optional keys; defaults are project=None, level=0.
    minimal = [
        {"name": "cybergym/bare", "module": "cybergym"},
    ]
    assert resolve_group("cybergym/level0", minimal) == ["cybergym/bare"]


# --- error cases ------------------------------------------------------------

def test_unknown_module_errors_with_known_modules(index):
    with pytest.raises(ValueError, match=r"unknown module 'cybergm'.*known modules"):
        resolve_group("cybergm/level1", index)


def test_unknown_level_errors_with_available_levels(index):
    with pytest.raises(ValueError, match="level 3"):
        resolve_group("cybergym/level3", index)


def test_unknown_project_errors_with_available_projects(index):
    with pytest.raises(ValueError, match="wget"):
        resolve_group("cybergym/project=wget", index)


def test_unknown_task_name_in_explicit_list(index):
    with pytest.raises(ValueError, match="cybergym/nope"):
        resolve_group(["cybergym/libxml2-lvl1", "cybergym/nope"], index)


def test_unknown_bare_string_errors(index):
    # A non-filter string (e.g. "cybergym/levelX") falls through to the
    # task-name path and errors identically.
    with pytest.raises(ValueError, match="cybergym/nope"):
        resolve_group("cybergym/nope", index)
    with pytest.raises(ValueError, match="cybergym/levelX"):
        resolve_group("cybergym/levelX", index)


def test_empty_explicit_list_errors(index):
    with pytest.raises(ValueError, match="empty"):
        resolve_group([], index)


def test_empty_index_errors_on_any_expression():
    with pytest.raises(ValueError, match="cybergym"):
        resolve_group("cybergym/level1", [])



def test_non_string_expression_raises_type_error(index):
    with pytest.raises(TypeError, match="string or sequence"):
        resolve_group(3, index)


def test_explicit_list_duplicates_pass_through(index):
    # Contract: explicit lists are returned verbatim, in input order —
    # duplicates are the caller's decision (e.g. task replicas), not an error.
    assert resolve_group(["cybergym/unclaimed", "cybergym/unclaimed"], index) == [
        "cybergym/unclaimed",
        "cybergym/unclaimed",
    ]


# --- index validation -------------------------------------------------------

def test_validate_index_accepts_valid_index(index):
    validate_index(index)  # no raise


def test_validate_index_rejects_duplicate_names():
    dup = [
        {"name": "a", "module": "m", "project": None, "level": 0},
        {"name": "a", "module": "m", "project": None, "level": 1},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_index(dup)


def test_validate_index_rejects_missing_required_key():
    with pytest.raises(ValueError, match="module"):
        validate_index([{"name": "a"}])


def test_validate_index_rejects_bad_types():
    with pytest.raises(ValueError, match="level"):
        validate_index([{"name": "a", "module": "m", "level": "one"}])
    with pytest.raises(ValueError, match="project"):
        validate_index([{"name": "a", "module": "m", "project": 3}])


def test_validate_index_rejects_non_mapping_entry():
    with pytest.raises(ValueError, match="not a mapping"):
        validate_index([["not-a-dict"]])


def test_resolve_group_rejects_duplicate_names_in_index():
    dup = [
        {"name": "a", "module": "m", "project": None, "level": 0},
        {"name": "a", "module": "m", "project": None, "level": 1},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_group("m/level0", dup)
