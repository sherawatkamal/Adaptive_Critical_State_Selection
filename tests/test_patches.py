"""Unit tests for patch apply: replace, insert, edit_query."""

import pytest

from idt.patches import (
    ReplaceActionPatch,
    InsertActionPatch,
    EditQueryPatch,
    Patch,
    is_search_action,
)


def test_replace_action_patch():
    actions = ["step", "stay", "step"]
    patch = ReplaceActionPatch(step_t=1, new_action="step")
    out = patch.apply(actions)
    assert out == ["step", "step", "step"]
    assert patch.cost() == 1.0


def test_replace_action_patch_to_dict_from_dict():
    patch = ReplaceActionPatch(step_t=0, new_action="step")
    d = patch.to_dict()
    assert d["patch_type"] == "replace"
    assert d["step_t"] == 0
    assert d["payload"] == "step"
    p2 = Patch.from_dict(d)
    assert p2.apply(["stay", "stay"]) == ["step", "stay"]


def test_insert_action_patch():
    actions = ["step", "step"]
    patch = InsertActionPatch(step_t=1, action_to_insert="stay")
    out = patch.apply(actions)
    assert out == ["step", "stay", "step"]
    assert patch.cost() == 2.0


def test_edit_query_patch():
    actions = ["search[shoes]", "click[item1]"]
    patch = EditQueryPatch(step_t=0, new_query="blue shoes")
    out = patch.apply(actions)
    assert out[0] == "search[blue shoes]"
    assert out[1] == "click[item1]"


def test_edit_query_patch_non_search_unchanged():
    actions = ["click[back]", "search[query]"]
    patch = EditQueryPatch(step_t=1, new_query="other")
    out = patch.apply(actions)
    assert out[0] == "click[back]"
    assert out[1] == "search[other]"


def test_is_search_action():
    assert is_search_action("search[foo]") is True
    assert is_search_action("click[bar]") is False
