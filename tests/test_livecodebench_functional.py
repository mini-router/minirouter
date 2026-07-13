"""Offline tests for LiveCodeBench call-based ("functional") scoring (#105).

Before this fix every functional problem scored 0: `fn_name` was read as a
top-level column (it lives in the `metadata` JSON), `testtype` was discarded,
and the grader only ran stdin/stdout programs. These tests cover the loader
(metadata `func_name`, preserved `testtype`) and the reward execution path
(call-based function invocation, tuple normalization) while asserting the
historical stdin path is untouched. Pure stdlib: they execute local Python in
the same isolated subprocess sandbox the grader uses (no network / GPU / torch).
"""
from __future__ import annotations

import json

import trinity.orchestration.dataset as D
import trinity.orchestration.reward as R

_TWO_SUM = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        seen = {}\n"
    "        for i, v in enumerate(nums):\n"
    "            if target - v in seen:\n"
    "                return [seen[target - v], i]\n"
    "            seen[v] = i\n"
)


def _fence(code: str) -> str:
    return "```python\n" + code + "\n```"


def _fspec(fn_name="twoSum", testtype="functional"):
    test = {"input": "[2,7,11,15]\n9", "output": "[0,1]"}
    if testtype is not None:
        test["testtype"] = testtype
    return {"tests": [test], "fn_name": fn_name}


def test_functional_solution_method_scores_correct():
    assert R.score_text("livecodebench", _fence(_TWO_SUM), _fspec()) == 1.0


def test_functional_wrong_answer_scores_zero():
    bad = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        return [9, 9]\n"
    )
    assert R.score_text("livecodebench", _fence(bad), _fspec()) == 0.0


def test_functional_tuple_return_normalizes_to_list():
    tup = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        return (0, 1)\n"
    )
    assert R.score_text("livecodebench", _fence(tup), _fspec()) == 1.0


def test_functional_module_level_function():
    spec = {"tests": [{"input": "5", "output": "25", "testtype": "functional"}], "fn_name": "square"}
    assert R.score_text("livecodebench", _fence("def square(n):\n    return n * n\n"), spec) == 1.0


def test_stray_fn_name_without_functional_testtype_uses_stdout():
    # testtype is the authoritative mode signal (review on #114): a stray fn_name
    # on a stdin/untyped case must NOT switch to the call-based harness, else a
    # normal stdin program would be graded call-based and falsely fail.
    spec = {"tests": [{"input": "3\n", "output": "9\n"}], "fn_name": "solve"}
    good = "import sys\nn=int(sys.stdin.read())\nprint(n*n)"
    assert R.score_text("livecodebench", _fence(good), spec) == 1.0
    # And a genuine functional case (explicit testtype) still grades call-based.
    assert R.score_text("livecodebench", _fence(_TWO_SUM), _fspec()) == 1.0


def test_stdin_problem_still_scored_by_stdout():
    spec = {
        "tests": [{"input": "3\n", "output": "9\n"}, {"input": "5\n", "output": "25\n"}],
        "fn_name": None,
    }
    good = "import sys\nn=int(sys.stdin.read())\nprint(n*n)"
    bad = "import sys\nn=int(sys.stdin.read())\nprint(n+n)"
    assert R.score_text("livecodebench", _fence(good), spec) == 1.0
    assert R.score_text("livecodebench", _fence(bad), spec) == 0.0


def test_explicit_stdin_testtype_uses_stdout_even_with_fn_name():
    # testtype == "stdin" forces the stdout path regardless of a stray fn_name.
    spec = {
        "tests": [{"input": "3\n", "output": "9\n", "testtype": "stdin"}],
        "fn_name": "irrelevant",
    }
    good = "import sys\nn=int(sys.stdin.read())\nprint(n*n)"
    assert R.score_text("livecodebench", _fence(good), spec) == 1.0


def test_loader_reads_func_name_from_metadata_json():
    row = {"metadata": json.dumps({"func_name": "foo"})}
    assert D._lcb_fn_name(row) == "foo"


def test_loader_reads_top_level_fn_name_fallback():
    assert D._lcb_fn_name({"fn_name": "bar"}) == "bar"


def test_loader_returns_none_for_stdin_problem():
    assert D._lcb_fn_name({"metadata": json.dumps({})}) is None
    assert D._lcb_fn_name({"fn_name": None}) is None


def test_parse_lcb_tests_preserves_testtype_when_present():
    row = {"public_test_cases": [{"input": "1", "output": "2", "testtype": "functional"}]}
    parsed = D._parse_lcb_tests(row)
    assert parsed == [{"input": "1", "output": "2", "testtype": "functional"}]


def test_parse_lcb_tests_omits_testtype_when_absent():
    row = {"public_test_cases": [{"input": "1", "output": "2"}]}
    assert D._parse_lcb_tests(row) == [{"input": "1", "output": "2"}]


def test_coerce_test_spec_extracts_fn_name():
    tests, timeout_s, fn_name = R._coerce_test_spec({"tests": [], "fn_name": "twoSum"})
    assert fn_name == "twoSum"
    tests, timeout_s, fn_name = R._coerce_test_spec({"tests": []})
    assert fn_name is None
