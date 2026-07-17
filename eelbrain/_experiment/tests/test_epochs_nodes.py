# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tests for eelbrain._experiment.epochs.nodes helpers."""
import numpy as np

from eelbrain._data_obj import Factor, Var
from eelbrain._experiment.epochs.nodes import _factor_trigger_to_var


def test_factor_trigger_to_var_is_numeric():
    "Result is a numeric Var, safe to use as an MNE (int32) event code"
    factor = Factor(['a', 'b', 'a', 'c'])
    var = _factor_trigger_to_var(factor)
    assert isinstance(var, Var)
    assert len(var) == len(factor)
    assert var.x.dtype.kind in 'iu'
    # fits in a signed int32 event-ID column
    assert (var.x >= 0).all()
    assert (var.x <= 2 ** 31 - 1).all()


def test_factor_trigger_to_var_same_label_same_code():
    "The same label always gets the same code within one Factor"
    factor = Factor(['a', 'b', 'a', 'c', 'b'])
    var = _factor_trigger_to_var(factor)
    by_label = {}
    for label, code in zip(factor, var.x):
        by_label.setdefault(label, set()).add(code)
    assert all(len(codes) == 1 for codes in by_label.values())
    # distinct labels never collide with each other here
    assert len({next(iter(codes)) for codes in by_label.values()}) == len(by_label)


def test_factor_trigger_to_var_stable_across_recordings():
    """A label's code does not depend on which other labels/values are
    present in that particular Factor -- i.e. it doesn't matter that Run B
    is missing some of the trigger values seen in Run A (e.g. no button
    presses happened to occur in that recording).
    """
    labels = {32: 'button', 5: 'smiley', 1: 'a', 2: 'b', 3: 'c', 4: 'd'}

    # Run A: all six triggers occur
    factor_a = Factor(Var(np.array([1, 2, 3, 4, 5, 32])), labels=labels)
    var_a = _factor_trigger_to_var(factor_a)
    code_a = dict(zip(factor_a, var_a.x))

    # Run B: only a subset of triggers occurs (e.g. no 'button', no 'd')
    factor_b = Factor(Var(np.array([1, 2, 3, 5])), labels=labels)
    var_b = _factor_trigger_to_var(factor_b)
    code_b = dict(zip(factor_b, var_b.x))

    for label in code_b:
        assert code_b[label] == code_a[label]


def test_factor_trigger_to_var_empty():
    "Empty Factor -> empty Var, no crash"
    factor = Factor([])
    var = _factor_trigger_to_var(factor)
    assert len(var) == 0
