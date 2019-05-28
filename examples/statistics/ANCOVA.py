"""
ANCOVA
======

Example 1
---------
Based on [1]_, `Exercises
<http://www.bio.ic.ac.uk/research/crawley/statistics/exercises/R6Ancova.pdf>`_
(page 8).
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *

y = Var([2, 3, 3, 4,
         3, 4, 5, 6,
         1, 2, 1, 2,
         1, 1, 2, 2,
         2, 2, 2, 2,
         1, 1, 2, 3], name="Growth Rate")

genotype = Factor(range(6), repeat=4, name="Genotype")

hours = Var([8, 12, 16, 24], tile=6, name="Hours")

###############################################################################
# Show the model
print(hours * genotype)

###############################################################################
# ANCOVA
print(test.anova(y, hours * genotype, title="ANOVA"))

###############################################################################
# Plot the slopes
plot.Regression(y, hours, genotype)


###############################################################################
# Example 2
# ---------
# Based on [2]_ (p. 118-20)

y = Var([16,  7, 11,  9, 10, 11,  8,  8,
         16, 10, 13, 10, 10, 14, 11, 12,
         24, 29, 10, 22, 25, 28, 22, 24])

cov = Var([9, 5, 6, 4, 6, 8, 3, 5,
           8, 5, 6, 5, 3, 6, 4, 6,
           5, 8, 3, 4, 6, 9, 4, 5], name='cov')

a = Factor([1, 2, 3], repeat=8, name='A')

###############################################################################
# Full model, with interaction
print(test.anova(y, cov * a))
plot.Regression(y, cov, a)

###############################################################################
# Drop interaction term
print(test.anova(y, a + cov))
plot.Regression(y, cov)


###############################################################################
# ANCOVA with multiple covariates
# -------------------------------
# Based on [3]_, p. 139.

# Load data form a text file
ds = load.txt.tsv('Fox_Prestige_data.txt', delimiter=None)
print(ds.head())

###############################################################################

# Variable summary
print(ds.summary())

###############################################################################

# Exclude cases with missing type
ds2 = ds[ds['type'] != 'NA']

# ANOVA
print(test.anova('prestige', '(income + education) * type', ds=ds2))


###############################################################################
# References
# ----------
# .. [1] Crawley, M. J. (2005). Statistics: an introduction using R. J Wiley.
# .. [2] Rutherford, A. (2001). Introducing ANOVA and ANCOVA: A GLM Approach. Sage.
# .. [3] Fox, J. (2008) Applied Regression Analysis and Generalized Linear Models, Second Edition. Sage.
