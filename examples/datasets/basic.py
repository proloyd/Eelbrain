"""
Dataset basics
==============
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *
import numpy as np

###############################################################################
# A dataset can be constructed column by column, by adding one variable after
# another:

# initialize an empty Dataset:
ds = Dataset()
# numeric values are added as Var object:
ds['y'] = Var(np.random.normal(0, 1, 6))
# categorical data as represented in Factors:
ds['a'] = Factor(['a', 'b', 'c'], repeat=2)
# check the result:
print(ds)

###############################################################################
# For larger datasets it can be more convenient to print only the first few
# cases...
print(ds.head())

###############################################################################
# ... or a summary of variables:
print(ds.summary())

###############################################################################
# A second way of constructing a dataset is case by case (i.e., row by row):

rows = []
for i in range(6):
    y = np.random.normal(0, 1)
    a = 'abc'[i % 3]
    rows.append([f'S{i}', y, a])
ds = Dataset.from_caselist(['subject', 'y', 'a'], rows)
print(ds)
