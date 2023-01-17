#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2020 Martine Lenders <m.lenders@fu-berlin.de>
#
# Distributed under terms of the MIT license.

import argparse
import fnmatch
import re
import os
import sys

import github

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def cs_string_list(value):  # pylint: disable=too-many-branches
    """
    Parses a comma separated strings into a python list

    # >>> cs_string_list('')
    # []
    # >>> cs_string_list("''")
    # ['']
    # >>> cs_string_list('""')
    # ['']
    # >>> cs_string_list('test')
    # ['test']
    # >>> cs_string_list('   test  ')
    # ['test']
    # >>> cs_string_list('   te"st"  ')
    # ['te"st"']
    # >>> cs_string_list('foo bar')
    # ['foo bar']
    # >>> cs_string_list("test,' foo bar'")
    # ['test', ' foo bar']
    >>> cs_string_list('"foo bar   "  ,    test   ')
    ['foo bar   ', 'test']
    >>> cs_string_list('test,   "' "'foo bar'" '"')
    ['test', "'foo bar'"]
    >>> cs_string_list("test, " '"bar, foo"' ",   'foo bar'    ")
    ['test', 'bar, foo', 'foo bar']
    >>> cs_string_list("test,,,")
    ['test', '', '', '']
    >>> cs_string_list("test,,,   ")
    ['test', '', '', '']
    """
    res = [""]
    in_str = False
    strip_str = True
    open_char = None
    last_char = None
    for char in value:
        if in_str:  # we are in a string
            if open_char:  # we are in a quoted string quoted with open_char
                if char == open_char:  # quoted string is closed
                    in_str = False
                    open_char = None
                else:
                    res[-1] += char
            elif char == ",":  # next string to process
                in_str = False
                # we only land here for unquoted strings as char == open_char
                # branch sets in_str == False
                res[-1] = res[-1].strip()
                res.append("")  # append next state
            else:
                res[-1] += char
        elif char == " ":  # skip non-quoted white-spaces
            continue
        elif char == ",":  # next string to process
            in_str = False
            if strip_str:  # was not a quoted string => strip
                res[-1] = res[-1].strip()
            else:
                strip_str = True  # else reset strip_str for next
            res.append("")  # append next state
        else:
            in_str = True  # we found a string!
            if char in ['"', "'"]:  # and it is quoted
                open_char = char
                strip_str = False
            else:
                res[-1] += char
        last_char = char
    if open_char:
        raise ValueError(f"Unterminated string '{res[-1]}'")
    if not res[-1] and last_char not in [",", "'", '"']:
        del res[-1]
    if res and strip_str:  # strip remaining unquoted string
        res[-1] = res[-1].strip()
    return res


def cs_2tuple_list(value):  # pylint: disable=too-many-branches
    """
    Parses a comma separated 2-tuple strings into a python list of tuples

    >>> cs_2tuple_list('')
    []
    >>> cs_2tuple_list('(foobar, "test")')
    [('foobar', 'test')]
    >>> cs_2tuple_list('(foobar, "test"),     ('"'barfoo',    "' lalala)   ')
    [('foobar', 'test'), ('barfoo', 'lalala')]
    >>> cs_2tuple_list('(foobar, "test"), ("(barfoo", "lalala)")')
    [('foobar', 'test'), ('(barfoo', 'lalala)')]
    """
    res = [""]
    in_tuple = False
    quote_char = None
    for char in value:
        if in_tuple:
            if not quote_char and char in ["'", '"']:
                quote_char = char
            elif char == quote_char:
                quote_char = None
            elif not quote_char and char == ")":
                res[-1] = tuple(cs_string_list(res[-1]))
                in_tuple = False
            else:
                res[-1] += char
        elif char == " ":
            continue
        elif char == ",":
            res.append("")
        elif char == "(":
            in_tuple = True
        else:
            raise ValueError(f"Unexpected character '{char}' after '{res}'")
    if in_tuple or quote_char:
        raise ValueError(f"Unterminated tuple {res[-1]}")
    # remove empty string stored as state
    if not isinstance(res[-1], tuple):
        del res[-1]
    if any(not isinstance(e, tuple) or len(e) != 2 for e in res):
        raise ValueError(f"Unexpected value in {res}")
    return res


def get_pull_no(ref):
    """
    Get pull request from a git given ref
    >>> get_pull_no('refs/pull/12345/head')
    12345
    >>> get_pull_no('refs/pull/6789/merge')
    6789
    """
    match = re.search("refs/pull/([0-9]+)/", ref)
    if match:
        return int(match[1])
    print(os.environ)
    if os.environ.get("INPUT_PULL_REQUEST"):
        return int(os.environ["INPUT_PULL_REQUEST"])
    raise ValueError(f"Unable to get pull request number from ref {ref}")


def parse_condition(condition):
    """
    Parses a condition for cond_labels

    >>> parse_condition("review.approval    > 3")
    ['review.approval', '3']
    """
    elems = condition.split(">")
    if len(elems) != 2:
        raise ValueError("Unable to parse ")
    return [e.strip() for e in elems]


def check_review_approvals(pull, condition, missing_approvals_label):
    condition[1] = int(condition[1])
    approvals = 0
    for review in pull.get_reviews():
        if review.state == "APPROVED":
            approvals += 1
            if approvals > condition[1]:
                print(
                    f"PR#{pull} has {approvals}/{condition[1] + 1} approvals,"
                    f" removing label '{missing_approvals_label}'"
                )
                if missing_approvals_label:
                    pull.remove_from_labels(missing_approvals_label)
                return True

    if missing_approvals_label and condition[1] > 0:
        print(
            f"PR#{pull} has only {approvals}/{condition[1] + 1} approvals,"
            f" setting label '{missing_approvals_label}'"
        )
        pull.add_to_labels(missing_approvals_label)

    return False


VALID_CONDITIONS = {"review.approvals": check_review_approvals}


def check_condition(pull, condition, missing_approvals_label):
    elems = parse_condition(condition)
    try:
        return VALID_CONDITIONS[elems[0]](pull, elems, missing_approvals_label)
    except KeyError:
        # We don't want the original traceback here
        # pylint: disable=raise-missing-from
        raise ValueError(f"Unrecognized condition {condition}")


def check_labels(set_labels, unset_labels, cond_labels, missing_approvals_label, pull):
    pull_labels = [label.name for label in pull.get_labels()]
    set_labels_check = [False for label in set_labels]
    for pull_label in pull_labels:
        for i, set_label in enumerate(set_labels):
            if fnmatch.fnmatch(pull_label, set_label):
                set_labels_check[i] = True
        for unset_label in unset_labels:
            if fnmatch.fnmatch(pull_label, unset_label):
                print(f"{', '.join(unset_labels)} are expected not to be set")
                return 1
    if not all(set_labels_check):
        print(f"{', '.join(set_labels)} are expected to be set")
        return 1

    res = 0
    for cond_label, condition in cond_labels:
        for pull_label in pull_labels:
            if fnmatch.fnmatch(pull_label, cond_label) and not check_condition(
                pull, condition, missing_approvals_label
            ):
                print(f"Condition {condition} for label {pull_label} not fulfilled")
                # favor listing all failed conditions over early exit
                res = 1
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "set_labels",
        default="",
        type=cs_string_list,
        help="Comma-separated list of labels required to be set. default: ''",
    )
    parser.add_argument(
        "unset_labels",
        default="",
        type=cs_string_list,
        help="Comma-separated list of labels required not to be set. default: ''",
    )
    parser.add_argument(
        "cond_labels",
        default="",
        type=cs_2tuple_list,
        help=(
            "Comma-separated list of (label, condition) for labels "
            "introducing a conditions. "
            "default: ''. "
            "Supported conditions: 'review.approvals>x' "
            "where x is a positive number"
        ),
    )
    parser.add_argument(
        "missing_approvals_label",
        default="",
        type=str,
        help="Name of label reflecting the approval status,"
        "will be set while approvals are missing"
        "default: '' (no label is managed). ",
    )
    args = parser.parse_args()

    repo_name = os.environ["GITHUB_REPOSITORY"]
    repo = github.Github(os.environ.get("INPUT_ACCESS_TOKEN", None)).get_repo(repo_name)

    if args.missing_approvals_label:
        # turn label string into label object
        try:
            missing_approvals_label = repo.get_label(args.missing_approvals_label)
        except github.GithubException:
            print(
                f"Error getting label '{args.missing_approvals_label}'"
                " from github. Does it exist?"
            )
            return 1
    else:
        missing_approvals_label = None

    return check_labels(
        args.set_labels,
        args.unset_labels,
        args.cond_labels,
        missing_approvals_label,
        repo.get_pull(get_pull_no(os.environ["GITHUB_REF"])),
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover


if pytest:  # noqa: C901

    @pytest.mark.parametrize(
        "value, exp",
        [("", [])],
    )
    def test_cs_string_list(value, exp):
        res = cs_string_list(value)
        assert res == exp

    @pytest.mark.parametrize(
        "value",
        ['"', "'", "test,' foo bar", 'test," foo bar', '"test', "'test"],
    )
    def test_cs_string_list_value_error(value):
        with pytest.raises(ValueError):
            cs_string_list(value)

    @pytest.mark.parametrize(
        "value",
        [")", '"', "'", "(test", '(test, foobar, "foo bar")'],
    )
    def test_cs_2tuple_list_value_error(value):
        with pytest.raises(ValueError):
            cs_2tuple_list(value)

    def test_get_pull_no_invalid():
        with pytest.raises(ValueError):
            get_pull_no("foobar")

    @pytest.mark.parametrize(
        "value",
        ["foobar", "foobar>3>test"],
    )
    def test_parse_condition_invalid(value):
        with pytest.raises(ValueError):
            parse_condition(value)

    # pylint: disable=too-few-public-methods
    class MockLabel:
        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            return self._name

    # pylint: disable=too-few-public-methods
    class MockReview:
        def __init__(self, state):
            self._state = state

        @property
        def state(self):
            return self._state

    class MockPull:
        def __init__(self, labels=None, reviews=None):
            self.labels = labels
            self.reviews = reviews

        def add_to_labels(self, label):
            if self.labels:
                self.labels.append(label)
            else:
                self.labels = [label]

        def remove_from_labels(self, label):
            if self.labels and label in self.labels:
                self.labels.remove(label)

        def get_labels(self):
            for label in self.labels:
                yield label

        def get_reviews(self):
            for review in self.reviews:
                yield review

    @pytest.mark.parametrize(
        "value, labels, reviews, missing_approvals_label, exp",
        [
            (
                ["review.approvals", 2],
                None,
                ["APPROVED", "APPROVED", "APPROVED"],
                "",
                True,
            ),
            (["review.approvals", 2], None, ["APPROVED", "APPROVED"], "", False),
            (["review.approvals", 1], None, ["APPROVED", "APPROVED"], "", True),
            (["review.approvals", 1], None, ["COMMENT", "APPROVED"], "", False),
            (["review.approvals", 1], None, ["COMMENT"], "", False),
            (["review.approvals", 1], None, ["APPROVED"], "DON'T MERGE", False),
            (["review.approvals", 1], ["FOOBAR"], ["APPROVED"], "DON'T MERGE", False),
            (
                ["review.approvals", 1],
                None,
                ["APPROVED", "APPROVED"],
                "DON'T MERGE",
                True,
            ),
            (
                ["review.approvals", 1],
                ["FOOBAR"],
                ["APPROVED", "APPROVED"],
                "DON'T MERGE",
                True,
            ),
            (
                ["review.approvals", 1],
                ["DON'T MERGE", "FOOBAR"],
                ["APPROVED", "APPROVED"],
                "DON'T MERGE",
                True,
            ),
        ],
    )
    def test_check_review_approvals(
        value, labels, reviews, missing_approvals_label, exp
    ):
        pull = MockPull(labels=labels, reviews=[MockReview(state) for state in reviews])
        assert check_review_approvals(pull, value, missing_approvals_label) == exp
        if not exp and missing_approvals_label:
            assert missing_approvals_label in pull.labels
        if exp and missing_approvals_label:
            assert not pull.labels or missing_approvals_label not in pull.labels

    def test_check_condition(monkeypatch):
        pull = MockPull()
        monkeypatch.setattr(
            sys.modules[__name__], "parse_condition", lambda x: ["test", 1]
        )
        monkeypatch.setitem(VALID_CONDITIONS, "test", lambda x, y, z: True)
        assert check_condition(pull, "test > 1", "")

    def test_check_condition_invalid(monkeypatch):
        pull = MockPull()
        monkeypatch.setattr(
            sys.modules[__name__], "parse_condition", lambda x: ["test", 1]
        )
        if "test" in VALID_CONDITIONS:
            monkeypatch.delitem(VALID_CONDITIONS, "test", lambda x, y: True)
        with pytest.raises(ValueError):
            check_condition(pull, "test > 1", "")

    @pytest.mark.parametrize(
        "set_labels, unset_labels, cond_labels, cond_res, pull_labels, "
        "missing_approvals_label, exp",
        [
            ([], [], [], False, [], "", 0),
            ([], [], [], False, ["lalala"], "", 0),
            ([], [], [], False, ["foobar"], "", 0),
            ([], [], [], True, [], "", 0),
            ([], [], [], True, ["lalala", "yes"], "", 0),
            ([], [], [], True, ["foobar"], "", 0),
            ([], [], [("foobar", "test>1")], False, [], "", 0),
            ([], [], [("foobar", "test>1")], False, ["lalala"], "", 0),
            ([], [], [("foobar", "test > 1")], False, ["lalala", "foobar"], "", 1),
            ([], [], [("foobar", "test>1")], True, [], "", 0),
            ([], [], [("foobar", "test>1")], True, ["lalala"], "", 0),
            ([], [], [("foobar", "test>1")], True, ["lalala", "foobar"], "", 0),
            ([], ["don't merge"], [], False, [], "", 0),
            ([], ["don't merge"], [], False, ["lalala"], "", 0),
            ([], ["don't merge"], [], False, ["lalala", "don't merge"], "", 1),
            ([], ["don't merge"], [], True, [], "", 0),
            ([], ["don't merge"], [], True, ["lalala"], "", 0),
            ([], ["don't merge"], [], True, ["lalala", "don't merge"], "", 1),
            ([], ["don't merge"], [("foobar", "test>1")], False, [], "", 0),
            ([], ["don't merge"], [("foobar", "test>1")], False, ["lalala"], "", 0),
            (
                [],
                ["don't merge"],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar"],
                "",
                1,
            ),
            (
                [],
                ["don't merge"],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar", "don't merge"],
                "",
                1,
            ),
            ([], ["don't merge"], [("foobar", "test>1")], True, [], "", 0),
            ([], ["don't merge"], [("foobar", "test>1")], True, ["lalala"], "", 0),
            (
                [],
                ["don't merge"],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar"],
                "",
                0,
            ),
            (
                [],
                ["don't merge"],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar", "don't merge"],
                "",
                1,
            ),
            (["yes"], [], [], False, [], "", 1),
            (["yes"], [], [], False, ["lalala"], "", 1),
            (["yes"], [], [], False, ["lalala", "don't merge"], "", 1),
            (["yes*"], [], [], False, ["lalala", "don't merge", "yes"], "", 0),
            (["yes"], [], [], True, [], "", 1),
            (["yes"], [], [], True, ["lalala"], "", 1),
            (["yes"], [], [], True, ["lalala", "don't merge"], "", 1),
            (["yes"], [], [], True, ["lalala", "don't merge", "yes"], "", 0),
            (["yes"], [], [("foobar", "test>1")], False, [], "", 1),
            (["yes"], [], [("foobar", "test>1")], False, ["lalala"], "", 1),
            (["yes"], [], [("foobar", "test>1")], False, ["lalala", "foobar"], "", 1),
            (["yes"], [], [("foobar", "test>1")], False, ["lalala", "yes"], "", 0),
            (
                ["yes"],
                [],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar", "don't merge"],
                "",
                1,
            ),
            (
                ["yes"],
                [],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar", "don't merge", "yes"],
                "",
                1,
            ),
            (["yes"], [], [("foobar", "test>1")], True, [], "", 1),
            (["yes"], [], [("foobar", "test>1")], True, ["lalala"], "", 1),
            (["yes"], [], [("foobar", "test>1")], True, ["lalala", "foobar"], "", 1),
            (
                ["yes"],
                [],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar", "don't merge"],
                "",
                1,
            ),
            (
                ["yes"],
                [],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar", "don't merge", "yes"],
                "",
                0,
            ),
            (["yes"], ["don't merge"], [], False, [], "", 1),
            (["yes"], ["don't merge"], [], False, ["lalala"], "", 1),
            (["yes"], ["don't merge"], [], False, ["lalala", "don't merge"], "", 1),
            (
                ["yes"],
                ["don't *ge"],
                [],
                False,
                ["lalala", "don't merge", "yes"],
                "",
                1,
            ),
            (["yes"], ["don't merge"], [], True, [], "", 1),
            (["yes"], ["don't merge"], [], True, ["lalala"], "", 1),
            (["yes"], ["don't *rge"], [], True, ["lalala", "don't merge"], "", 1),
            (["yes"], ["don't merge"], [], True, ["lalala", "yes"], "", 0),
            (
                ["yes"],
                ["don't merge"],
                [],
                True,
                ["lalala", "don't merge", "yes"],
                "",
                1,
            ),
            (["ye*"], ["don't merge"], [("foobar", "test>1")], False, [], "", 1),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                False,
                ["lalala"],
                "",
                1,
            ),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar"],
                "",
                1,
            ),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar", "don't merge"],
                "",
                1,
            ),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar", "don't merge", "yes"],
                "",
                1,
            ),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                False,
                ["lalala", "foobar", "yes"],
                "",
                1,
            ),
            (["yes"], ["don't merge"], [("foobar", "test>1")], True, [], "", 1),
            (["yes"], ["don't merge"], [("foobar", "test>1")], True, ["lalala"], "", 1),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar"],
                "",
                1,
            ),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar", "don't merge"],
                "",
                1,
            ),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar", "yes"],
                "",
                0,
            ),
            (
                ["yes"],
                ["don't merge"],
                [("foobar", "test>1")],
                True,
                ["lalala", "foobar", "don't merge", "yes"],
                "",
                1,
            ),
        ],
    )
    # pylint: disable=too-many-arguments
    def test_check_labels(
        monkeypatch,
        set_labels,
        unset_labels,
        cond_labels,
        cond_res,
        pull_labels,
        missing_approvals_label,
        exp,
    ):
        pull = MockPull(labels=[MockLabel(name) for name in pull_labels])
        monkeypatch.setattr(
            sys.modules[__name__], "check_condition", lambda x, y, z: cond_res
        )
        assert (
            check_labels(
                set_labels, unset_labels, cond_labels, missing_approvals_label, pull
            )
            == exp
        )

    @pytest.mark.parametrize(
        "missing_approvals_label, get_label_errors, exp",
        [
            pytest.param("", False, 0, id="no missing_approvals_label"),
            pytest.param("DON'T MERGE", False, 0, id="with missing_approvals_label"),
            pytest.param("DON'T MERGE", True, 1, id="with missing_approvals_label"),
            pytest.param("DON'T MERGE", False, 0, id="with missing_approvals_label"),
        ],
    )
    def test_main(monkeypatch, missing_approvals_label, get_label_errors, exp):
        class MockArgs:
            set_labels = ["yes"]
            unset_labels = []
            cond_labels = []
            missing_approvals_label = ""

        class MockRepo:
            def get_pull(self, pull_no):
                return pull_no

            def get_label(self, label):
                raise github.GithubException(status=404, data="")

        class MockGithub:
            def __init__(self, *args, **kwargs):
                pass

            # pylint: disable=unused-argument
            def get_repo(self, name):
                return MockRepo()

        MockArgs.missing_approvals_label = missing_approvals_label
        if not get_label_errors:
            MockRepo.get_label = lambda self, label: label
        monkeypatch.setattr(
            argparse.ArgumentParser, "add_argument", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            argparse.ArgumentParser, "parse_args", lambda *args, **kwargs: MockArgs()
        )
        monkeypatch.setattr(github, "Github", lambda *args, **kwargs: MockGithub())
        monkeypatch.setattr(
            sys.modules[__name__], "check_labels", lambda *args, **kwargs: 0
        )
        monkeypatch.setattr(
            sys.modules[__name__], "get_pull_no", lambda *args, **kwargs: 12345
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "test")
        monkeypatch.setenv("GITHUB_REF", "foobar")
        assert main() == exp
