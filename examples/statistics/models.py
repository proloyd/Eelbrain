"""
Model coding
============

Illustrates how to inspect coding of regression models.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *

ds = Dataset()
ds['A'] = Factor(['a1', 'a0'], repeat=4)
ds['B'] = Factor(['b1', 'b0'], repeat=2, tile=2)

###############################################################################
# look at data
ds.head()

###############################################################################
# create a fixed effects model
m = ds.eval('A * B')
print(repr(m))
###############################################################################
# show the model coding
print(m)

###############################################################################
# create random effects model
ds['subject'] = Factor(['s1', 's2'], tile=4, name='subject', random=True)
m = ds.eval('A * B * subject')
print(repr(m))
###############################################################################
# show the model coding
print(m)
