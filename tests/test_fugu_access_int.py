"""Offline unit tests for bare-int access indices in fugu workflow parsing.

Regression for: `_normalize_access` accepted a scalar string digit (`"0"` -> `[0]`)
and `None` (-> `[]`) but rejected a bare scalar int (`0`), dropping the whole
workflow (`parsed_ok=False`) even though a bare index is the most natural model
output and is semantically identical to the accepted forms. Pure functions; no GPU.
"""
from trinity.fugu.workflow import _normalize_access, parse_workflow


def test_bare_int_access_index_accepted():
    txt = 'model_id=[0,1]\nsubtasks=["solve","answer"]\naccess_list=[[], 0]'
    wf, ok = parse_workflow(txt, n_workers=3)
    assert ok and wf is not None
    assert wf.steps[1].access == [0]


def test_bare_int_matches_string_and_list_forms():
    base = 'model_id=[0,1]\nsubtasks=["solve","answer"]\naccess_list=[[], {}]'
    outs = []
    for form in ("0", '"0"', "[0]"):
        wf, ok = parse_workflow(base.format(form), n_workers=3)
        assert ok, f"form {form} should parse"
        outs.append(wf.steps[1].access)
    assert outs[0] == outs[1] == outs[2] == [0]


def test_normalize_access_int_directly():
    assert _normalize_access(0, 1) == [0]
    assert _normalize_access(2, 3) == [2]
    # forward / out-of-range reference is invalid (rejects), same as the list form
    assert _normalize_access(3, 3) is None
    assert _normalize_access(-1, 3) is None


def test_bool_is_not_a_valid_access_index():
    # bool is a subclass of int, but True/False are not step indices
    assert _normalize_access(True, 3) is None
    assert _normalize_access(False, 3) is None
