"""Offline tests for LiveCodeBench functional (call-based) grading.

LiveCodeBench problems come in two flavors, tagged by ``testtype``:

* ``stdin``      - the candidate is run as a program, fed stdin, stdout compared.
* ``functional`` - the candidate defines ``Solution.<func_name>`` (or a module-level
  ``<func_name>``) which must be **called** with JSON-decoded arguments.

The function name lives inside LiveCodeBench's JSON ``metadata`` blob, not a
top-level column. Before this was wired up, every functional problem was executed
as a stdin program and therefore scored 0 no matter how correct the solution was.

No network / no GPU: the reward checker runs candidate code in a subprocess.
"""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R
from trinity.orchestration.dataset import _lcb_func_name, _parse_lcb_tests

_SOLUTION_CLASS = """
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, v in enumerate(nums):
            if target - v in seen:
                return [seen[target - v], i]
            seen[v] = i
"""

_MODULE_LEVEL = """
def twoSum(nums, target):
    seen = {}
    for i, v in enumerate(nums):
        if target - v in seen:
            return [seen[target - v], i]
        seen[v] = i
"""

_WRONG = """
class Solution:
    def twoSum(self, nums, target):
        return [0, 0]
"""

_FUNCTIONAL_SPEC = {
    "tests": [
        {"input": "[2,7,11,15]\n9", "output": "[0,1]", "testtype": "functional"},
        {"input": "[3,2,4]\n6", "output": "[1,2]", "testtype": "functional"},
    ],
    "fn_name": "twoSum",
}


def _fenced(code: str) -> str:
    return f"```python\n{code}\n```"


# --------------------------------------------------------------------------- #
# reward: functional execution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code", [_SOLUTION_CLASS, _MODULE_LEVEL])
def test_correct_functional_solution_scores_one(code):
    # The regression: this used to score 0.0 because fn_name was ignored and the
    # candidate was run as a stdin program.
    assert R.score_text("livecodebench", _fenced(code), _FUNCTIONAL_SPEC) == 1.0


def test_wrong_functional_solution_scores_zero():
    assert R.score_text("livecodebench", _fenced(_WRONG), _FUNCTIONAL_SPEC) == 0.0


def test_functional_requires_all_tests_to_pass():
    # Passes the first case only.
    partial = """
class Solution:
    def twoSum(self, nums, target):
        return [0, 1]
"""
    assert R.score_text("livecodebench", _fenced(partial), _FUNCTIONAL_SPEC) == 0.0


def test_missing_function_name_scores_zero():
    code = "class Solution:\n    def somethingElse(self):\n        return 1\n"
    assert R.score_text("livecodebench", _fenced(code), _FUNCTIONAL_SPEC) == 0.0


def test_tuple_result_matches_list_expected():
    code = "def twoSum(nums, target):\n    return (0, 1)\n"
    spec = {
        "tests": [{"input": "[2,7,11,15]\n9", "output": "[0,1]", "testtype": "functional"}],
        "fn_name": "twoSum",
    }
    assert R.score_text("livecodebench", _fenced(code), spec) == 1.0


# --------------------------------------------------------------------------- #
# reward: stdin flavor and legacy behavior must be untouched
# --------------------------------------------------------------------------- #
_SQUARE = "n = int(input())\nprint(n * n)\n"


def test_stdin_tests_still_work():
    spec = {
        "tests": [
            {"input": "3\n", "output": "9", "testtype": "stdin"},
            {"input": "5\n", "output": "25", "testtype": "stdin"},
        ]
    }
    assert R.score_text("livecodebench", _fenced(_SQUARE), spec) == 1.0


def test_without_fn_name_dict_tests_are_stdin_legacy():
    # No fn_name -> historical behavior (input/output treated as stdin/stdout).
    spec = {"tests": [{"input": "3\n", "output": "9"}]}
    assert R.score_text("livecodebench", _fenced(_SQUARE), spec) == 1.0


def test_assert_tests_still_work():
    code = "def add(a, b):\n    return a + b\n"
    assert R.score_text("livecodebench", _fenced(code), ["assert add(1, 2) == 3"]) == 1.0


def test_empty_tests_are_unscoreable():
    assert R.score_text("livecodebench", _fenced(_SQUARE), {"tests": []}) == 0.0


# --------------------------------------------------------------------------- #
# reward: helpers
# --------------------------------------------------------------------------- #
def test_parse_functional_args_decodes_one_json_value_per_line():
    assert R._parse_functional_args("[2,7,11,15]\n9") == [[2, 7, 11, 15], 9]
    assert R._parse_functional_args('  "abc" \n\n 3 ') == ["abc", 3]
    assert R._parse_functional_args("not json") is None


def test_coerce_test_spec_surfaces_fn_name():
    tests, timeout_s, fn_name = R._coerce_test_spec(_FUNCTIONAL_SPEC)
    assert fn_name == "twoSum"
    assert len(tests) == 2
    assert timeout_s == 10
    # A bare list of tests carries no fn_name.
    assert R._coerce_test_spec(["assert True"])[2] is None


def test_is_functional_test_dispatch():
    assert R._is_functional_test({"testtype": "functional"}, "f") is True
    assert R._is_functional_test({"testtype": "stdin"}, "f") is False
    assert R._is_functional_test({"input": "1", "output": "1"}, "f") is True
    assert R._is_functional_test({"stdin": "1"}, "f") is False
    # Without a fn_name nothing is functional.
    assert R._is_functional_test({"testtype": "functional"}, None) is False


# --------------------------------------------------------------------------- #
# dataset: func_name lives in the JSON metadata blob, testtype is preserved
# --------------------------------------------------------------------------- #
def test_lcb_func_name_read_from_metadata_json():
    assert _lcb_func_name({"metadata": '{"func_name": "twoSum"}'}) == "twoSum"
    assert _lcb_func_name({"metadata": {"func_name": "maxProfit"}}) == "maxProfit"
    # stdin problems carry no func_name
    assert _lcb_func_name({"metadata": "{}"}) is None
    assert _lcb_func_name({"metadata": "not json"}) is None
    assert _lcb_func_name({}) is None
    # a mirror exposing a real top-level column still works
    assert _lcb_func_name({"fn_name": "solve"}) == "solve"


def test_parse_lcb_tests_preserves_testtype():
    row = {
        "public_test_cases": (
            '[{"input": "[1,2]\\n3", "output": "[0,1]", "testtype": "functional"},'
            ' {"input": "3\\n", "output": "9", "testtype": "stdin"}]'
        )
    }
    tests = _parse_lcb_tests(row)
    assert [t["testtype"] for t in tests] == ["functional", "stdin"]
    assert tests[0]["input"] == "[1,2]\n3"


def test_parse_lcb_tests_defaults_to_stdin():
    row = {"public_test_cases": '[{"input": "1", "output": "1"}]'}
    assert _parse_lcb_tests(row)[0]["testtype"] == "stdin"
