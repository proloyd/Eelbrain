"""
Align datasets
==============

Shows how to combine information from two datasets describing the same cases,
but not necessarily in the same order.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import random
import string

from eelbrain import *


# Generate a dataset with known sequence
ds = Dataset()
ds['ascii'] = Factor(string.ascii_lowercase)
# Add an index variable to the dataset to later identify the cases
ds.index()

# Generate two shuffled copies of the dataset (and print them to confirm that
# they are shuffled)
ds1 = ds[random.sample(range(ds.n_cases), 15)]
print(ds1.head())
ds2 = ds[random.sample(range(ds.n_cases), 16)]
print(ds2.head())


###############################################################################
# Align the datasets
# ------------------
#
# Use the ``"index"`` variable added above to identify cases and align the two
# datasets

ds1_aligned, ds2_aligned = align(ds1, ds2, 'index')

# show the ascii sequences for the two datasets next to each other to
# demonstrate that they are aligned
ds1_aligned['ascii_ds2'] = ds2_aligned['ascii']
print(ds1_aligned)
