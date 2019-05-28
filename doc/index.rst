.. Building: $ make html
   If you get ValueError: unknown locale: UTF-8, run 
   export LC_ALL=en_US.UTF-8 
   before make (http://readthedocs.org/docs/flightdeck/en/latest/installation.html).

Eelbrain
========

.. image:: https://zenodo.org/badge/3651023.svg
   :target: https://zenodo.org/badge/latestdoi/3651023

.. image:: https://img.shields.io/conda/vn/conda-forge/eelbrain.svg
   :target: https://github.com/christianbrodbeck/Eelbrain/wiki/Installing

.. image:: https://img.shields.io/conda/pn/conda-forge/eelbrain.svg
   :target: https://anaconda.org/conda-forge/eelbrain

Eelbrain is an open-source `Python <https://www.python.org>`_ package for
accessible statistical analysis of MEG and EEG data.
It is maintained by
`Christian Brodbeck <http://christianbrodbeck.net>`_
at the `Computational sensorimotor systems lab
<http://www.isr.umd.edu/Labs/CSSL/simonlab/Home.html>`_
at University of Maryland, College Park.

If you use Eelbrain in work that is published, please acknowledge it by citing
it with the appropriate version and DOI.


Manual
------

.. toctree::
   :maxdepth: 1

   installing
   getting_started
   intro
   changes
   publications
   development

.. toctree::
   :maxdepth: 2

   reference
   auto_examples/index
   recipes
   experiment


.. seealso::
   - `Wiki <https://github.com/christianbrodbeck/Eelbrain/wiki>`_ on GitHub
   - `Mailing list <https://groups.google.com/forum/#!forum/eelbrain>`_ for announcements
   - Source code on `GitHub <http://github.com/christianbrodbeck/Eelbrain>`_
   - Eelbrain on the `Python Package Index <https://pypi.python.org/pypi/eelbrain>`_


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Eelbrain relies on
`NumPy <http://www.numpy.org>`_,
`SciPy <http://scipy.org>`_,
`Matplotlib <http://matplotlib.org>`_,
`MNE-Python <http://martinos.org/mne/stable/index.html>`_,
`PySurfer <http://pysurfer.github.io>`_,
`WxPython <http://wxpython.org>`_,
`Cython <http://cython.org>`_ and incorporates icons from the
`Tango Desktop Project <http://tango.freedesktop.org>`_.


----

Current funding: `National Institutes of Health (NIH) <https://www.nih.gov>`_
grant R01-DC-014085 (since 2016).
Past funding: `NYU Abu Dhabi Institute
<http://nyuad.nyu.edu/en/research/nyuad-institute.html>`_ grant G1001
(2011-2016).
