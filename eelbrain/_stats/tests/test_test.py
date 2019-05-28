# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from numpy.testing import assert_array_equal
import numpy as np
import pytest
import scipy.stats

from eelbrain import datasets, test
from eelbrain.fmtxt import asfmtext
from eelbrain._stats import test as _test


def test_correlations():
    "Test test.correlations()"
    ds = datasets.get_uv()

    res = test.correlations('fltvar', 'fltvar2', ds=ds)
    print(res)
    assert str(res[2][0]).strip() == '.398'
    res = test.correlations('fltvar', 'fltvar2', ds=ds, asds=True)
    assert res[0, 'r'] == pytest.approx(.398, abs=1e-3)
    res = test.Correlation('fltvar', 'fltvar2', ds=ds)
    assert res.r == pytest.approx(.398, abs=1e-3)

    res = test.correlations('fltvar', 'fltvar2', 'A', ds=ds)
    print(res)
    assert str(res[2][0]).strip() == 'a1'
    assert str(res[2][1]).strip() == '-.149'
    assert str(res[3][1]).strip() == '.740'
    res = test.correlations('fltvar', 'fltvar2', 'A', ds=ds, asds=True)
    assert res[0, 'A'] == 'a1'
    assert res[0, 'r'] == pytest.approx(-0.149, abs=1e-3)
    assert res[1, 'r'] == pytest.approx(.740, abs=1e-3)
    res = test.Correlation('fltvar', 'fltvar2', "A == 'a1'", ds)
    assert res.r == pytest.approx(-0.149, abs=1e-3)

    res = test.correlations('fltvar', 'fltvar2', 'A%B', ds=ds)
    print(res)
    assert str(res[2][2]).strip() == '-.276'
    res = test.correlations('fltvar', 'fltvar2', 'A%B', ds=ds, asds=True)
    assert res[0, 'r'] == pytest.approx(-0.276, abs=1e-3)

    res = test.correlations('fltvar', ('fltvar2', 'intvar'), 'A%B', ds=ds)
    print(res)
    assert str(res[2][1]).strip() == 'a1'
    assert str(res[2][2]).strip() == 'b1'
    assert str(res[2][3]).strip() == '-.276'
    res = test.correlations('fltvar', ('fltvar2', 'intvar'), 'A%B', ds=ds, asds=True)
    assert res[0, 'r'] == pytest.approx(-0.276, abs=1e-3)
    res = test.Correlation('fltvar', 'intvar', "(A=='a1')&(B=='b1')", ds)
    assert res.r == pytest.approx(0.315, abs=1e-3)


def test_star():
    "Test the star function"
    assert_array_equal(_test.star([0.1, 0.04, 0.01], int), [0, 1, 2])
    assert_array_equal(_test.star([0.001, 0.04, 0.1], int), [3, 1, 0])


def test_ttest():
    """Test univariate t-test functions"""
    ds = datasets.get_uv()

    print(test.ttest('fltvar', ds=ds))
    print(test.ttest('fltvar', 'A', ds=ds))
    print(test.ttest('fltvar', 'A%B', ds=ds))
    print(test.ttest('fltvar', 'A', match='rm', ds=ds))
    print(test.ttest('fltvar', 'A', 'a1', match='rm', ds=ds))
    print(test.ttest('fltvar', 'A%B', ('a1', 'b1'), match='rm', ds=ds))

    # Prepare data for scipy
    a1_index = ds.eval("A == 'a1'")
    a2_index = ds.eval("A == 'a2'")
    b1_index = ds.eval("B == 'b1'")
    a1_in_b1_index = np.logical_and(a1_index, b1_index)
    a2_in_b1_index = np.logical_and(a2_index, b1_index)

    # TTest1Samp
    res = test.TTest1Sample('fltvar', ds=ds)
    t, p = scipy.stats.ttest_1samp(ds['fltvar'], 0)
    assert res.t == pytest.approx(t, 10)
    assert res.p == pytest.approx(p, 10)
    res = test.TTest1Sample('fltvar', ds=ds, tail=1)
    assert res.t == pytest.approx(t, 10)
    assert res.p == pytest.approx(p / 2., 10)

    # TTestInd
    res = test.TTestInd('fltvar', 'A', 'a1', 'a2', ds=ds)
    t, p = scipy.stats.ttest_ind(ds[a1_index, 'fltvar'], ds[a2_index, 'fltvar'])
    assert res.t == pytest.approx(t, 10)
    assert res.p == pytest.approx(p, 10)

    # TTestRel
    res = test.TTestRel('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds)
    a1 = ds[a1_in_b1_index, 'fltvar'].x
    a2 = ds[a2_in_b1_index, 'fltvar'].x
    difference = a1 - a2
    t, p = scipy.stats.ttest_rel(a1, a2)
    assert_array_equal(res.difference.x, difference)
    assert res.df == len(a1) - 1
    assert res.tail == 0
    assert res.t == pytest.approx(t)
    assert res.p == pytest.approx(p)
    print(res)
    print(asfmtext(res))

    res = test.TTestRel('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds, 1)
    assert_array_equal(res.difference.x, difference)
    assert res.df == len(a1) - 1
    assert res.tail == 1
    assert res.t == pytest.approx(t)
    assert res.p == pytest.approx(p / 2)
    print(res)
    print(asfmtext(res))

    res = test.TTestRel('fltvar', 'A', 'a2', 'a1', 'rm', "B=='b1'", ds, 1)
    assert_array_equal(res.difference.x, -difference)
    assert res.df == len(a1) - 1
    assert res.tail == 1
    assert res.t == pytest.approx(-t)
    assert res.p == pytest.approx(1 - p / 2)

    res = test.TTestRel('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds, -1)
    assert_array_equal(res.difference.x, difference)
    assert res.df == len(a1) - 1
    assert res.tail == -1
    assert res.t == pytest.approx(t)
    assert res.p == pytest.approx(1 - p / 2)
    print(res)
    print(asfmtext(res))
    # alternative argspec
    a1 = ds.eval("fltvar[(B == 'b1') & (A == 'a1')]")
    a2 = ds.eval("fltvar[(B == 'b1') & (A == 'a2')]")
    res_alt = test.TTestRel(a1, a2, tail=-1)
    print(res_alt)
    assert res_alt.t == res.t
    assert res_alt.p == res.p
