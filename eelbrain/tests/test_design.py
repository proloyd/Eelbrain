import numpy as np
import pytest

from eelbrain import Dataset, Factor
from eelbrain._design import permute, random_factor, complement


def test_random_factor():
    """Test the design module for creating an experiemnt design"""
    ds = permute((
                  ('A', '123456'),
                  ('Bin', '01'),
                  ('B', 'abcdef'),
                  ))
    n = ds.n_cases

    rand = random_factor(('1', '2', '3'), n, 'rand')
    nv = (rand == '1').sum()
    assert nv == (rand == '2').sum(), "overall balancing"
    assert nv == (rand == '3').sum(), "overall balancing"

    # test urn kwarg
    randu = random_factor(('1', '2', '3'), n, urn=[rand])
    assert (rand == randu).sum() == 0, "`urn` arg failed"
    nv = (randu == '1').sum()
    assert nv == 24, "random value assignment"
    assert nv == (randu == '2').sum(), "overall balancing"
    assert nv == (randu == '3').sum(), "overall balancing"

    # test sub kwarg
    sub = ds['Bin'] == '1'
    subrand = random_factor(('1', '2', '3'), n, urn=[rand], sub=sub)
    assert np.all(rand != randu), "`urn` arg failed with `sub` arg"
    subc = (sub == False)
    assert np.all(subrand[subc] == ''), "values outside of sub are not ''"
    nv = (subrand == '1').sum()
    assert nv == 12, "random value assignment with `sub` arg"
    assert nv == (subrand == '2').sum(), "sub balancing"
    assert nv == (subrand == '3').sum(), "sub balancing"


def test_complement():
    """Test design.complement()"""
    ds = Dataset()
    ds['A'] = Factor('abcabc')
    ds['B'] = Factor('bcabca')
    ds['C'] = Factor('cabcab')

    # underspecified
    with pytest.raises(ValueError):
        complement(['A'], ds=ds)

    # correct
    comp = complement(['A', 'B'], ds=ds)
    assert np.all(comp == ds['C']), f"Complement yielded {comp} instead of {ds['C']}"

    # overspecified
    with pytest.raises(ValueError):
        complement(['A', 'B', 'C'], ds=ds)
