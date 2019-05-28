***********
Development
***********

Eelbrain is actively developed and maintained by
`Christian Brodbeck <http://loop.frontiersin.org/people/120185>`_
at the `Computational sensorimotor systems lab
<http://www.isr.umd.edu/Labs/CSSL/simonlab/Home.html>`_
at University of Maryland, College Park.

Eelbrain is fully open-source and new contributions are welcome on
`GitHub <https://github.com/christianbrodbeck/Eelbrain>`_. Suggestions can be
raised as issues, and modifications can be made as pull requests into the
``master`` branch.


The Development Version
-----------------------

The Eelbrain source code is hosted on
`GitHub <https://github.com/christianbrodbeck/Eelbrain>`_. Development takes
place on the ``master`` branch, while release versions are maintained on
branches called ``r/0.26`` etc. For further information on working with
GitHub see
`GitHub's instructions <https://help.github.com/articles/fork-a-repo/>`_.

Installing the development version requires the presence of a compiler.
On macOS, make sure Xcode is installed (open it once to accept the license
agreement).
Windows will indicate any needed files when the install command is run.

After cloning the repository, the development version can be installed by
running, from the ``Eelbrain`` repository's root directory::

    $ python setup.py develop


On macOS, the ``$ eelbrain`` shell script to run ``iPython`` with the framework
build is not installed properly by ``setup.py``; in order to fix this, run::

    $ ./fix-bin


In Python, you can make sure that you are working with the development version::

    >>> import eelbrain
    >>> eelbrain.__version__
    'dev'

To switch back to the release version use ``$ pip uninstall eelbrain``.


Building with Conda
-------------------

To build Eelbrain with ``conda``, make sure that  ``conda-build`` is installed.
Then, from ``Eelbrain/conda`` run::

    $ conda build eelbrain

After building successfully, the build can be installed with::

    $ conda install --use-local eelbrain


Contributing
------------

Style guides:

- Python: `PEP8 <https://www.python.org/dev/peps/pep-0008>`_
- Documentation: `NumPy documentation style
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
- `ReStructured Text Primer <http://sphinx-doc.org/rest.html>`_


Useful tools:

- Graphical frontend for git: `SourceTree <https://www.sourcetreeapp.com>`_
- Python IDE: `PyCharm <https://www.jetbrains.com/pycharm>`_


Testing
-------

Tests for individual modules are included in folders called ``tests``, usually
on the same level as the module.
To run all tests, run ``$ make test`` from the Eelbrain project directory.
On macOS, tests needs to run with the framework build of Python;
if you get a corresponding error, run ``$ ./fix-bin pytest`` from the
``Eelbrain`` repository root.
