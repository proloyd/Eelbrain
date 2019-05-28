Changes
=======

.. currentmodule:: eelbrain

New in 0.31
-----------

* :mod:`testnd` API:  For permutation tests, the ``samples`` parameter now defaults to 10,000 (previously 0)


New in 0.30
-----------

* Support for vector data (with many contributions from `Proloy Das`_):

  - :class:`Space` dimension to represent physical space
  - :class:`VolumeSourceSpace` to represent volume source spaces
  - Statistical tests: :class:`testnd.Vector`, :class:`testnd.VectorDifferenceRelated`
  - Plotting with :class:`plot.GlassBrain`

* ICA-GUI: tool to find high amplitude signals
* Documentation: New :ref:`examples` section
* :meth:`Dataset.summary` method
* .. currentmodule:: eelbrain.pipeline
  :class:`MneExperiment` pipeline:

  - :class:`RawApplyICA` preprocessing pipe to apply ICA estimated in a different branch of the pipeline.
  - :meth:`MneExperiment.load_evoked_stc` API more closely matches :meth:`MneExperiment.load_epochs_stc`
  - :meth:`MneExperiment.load_neighbor_correlation`



New in 0.29
-----------

.. currentmodule:: eelbrain

* API: Better default parameters for :func:`resample`
* Predictor-specific stopping for :func:`boosting`
* New :class:`NDVar` function :func:`correlation_coefficient`
* Plotting:

  - :ref:`general-layout-parameters` for plot size relative to screen size
  - Better plots for masked statistic maps

* .. currentmodule:: eelbrain.pipeline
  :class:`MneExperiment` pipeline:

  - API: :meth:`MneExperiment.make_rej` renamed to :meth:`MneExperiment.make_epoch_selection`
  - Object-based definitions (see :ref:`experiment-class-guide`)
  - New method :meth:`MneExperiment.plot_raw`


New in 0.28
-----------

.. currentmodule:: eelbrain

* Transition to `Python 3.6 <https://docs.python.org/3.6/>`_
* API changes:

  - :class:`testnd.anova`: The ``match`` parameter is now determined
    automatically and does not need to be specified anymore in most cases.
  - :attr:`testnd.ttest_1samp.diff` renamed to
    :attr:`testnd.ttest_1samp.difference`.
  - :class:`plot.Histogram`: following :mod:`matplotlib`, the ``normed``
    parameter was renamed to ``density``.
  - Previously capitalized argument and attribute names ``Y``, ``X`` and ``Xax``
    are now lowercase.
  - Topomap-plot argument order changed to provide consistency between different
    plots.

* :class:`NDVar` and :class:`Var` support for ``round(x)``
* .. currentmodule:: eelbrain.pipeline
  :class:`MneExperiment` pipeline:

  - Independent measures t-test


New in 0.27
-----------

.. currentmodule:: eelbrain

* API changes:

  - To change the parcellation of an :class:`NDVar` with source-space data,
    use the new function :func:`set_parc`. The :meth:`SourceSpace.set_parc`
    method has been removed because dimension objects should be treated
    as immutable, as they can be shared between different :class:`NDVar`
    instances. Analogously, :meth:`UTS.set_tmin` is now :func:`set_tmin`.
  - :func:`table.frequencies`: If the input ``y`` is a :class:`Var` object, the
    output will also be a :class:`Var` (was :class:`Factor`).
  - :meth:`NDVar.smooth`: window-based smoothing now uses a symmetric window,
    which can lead to slightly different results.

* :func:`concatenate`: concatenate multiple :class:`NDVar` objects to form a
  new dimension.
* :meth:`NDVar.ols`: regress on a dimension.
* :class:`plot.brain.SequencePlotter` to plot multiple anatomical images on one
  figure.
* New functions and objects:

  - :func:`complete_source_space`
  - :func:`psd_welch`
  - :func:`frequency_response`
  - :class:`test.Correlation`, :func:`test.pairwise_correlations`
  - :func:`xhemi`

* New methods:

  - :meth:`Dataset.zip`, :meth:`Dataset.tile`
  - :meth:`Factor.sort_cells`


New in 0.26
-----------

* API changes:

  - A new global :func:`configure` function replaces module-level configuration
    functions.
  - :class:`Dataset`: when a one-dimensional array is assigned to an unused
    key, the array is now automatically converted to a :class:`Var` object.
  - :attr:`SourceSpace.vertno` has been renamed to
    :attr:`SourceSpace.vertices`.

* Plotting:

  - The new ``name`` argument allows setting the window title without adding a
    title to the figure.
  - Plots that reresent time have a new method to synchronize the time axis on
    multiple plots: :meth:`~plot.Butterfly.link_time_axis`.
  - Plot source space time series: :func:`plot.brain.butterfly`

* ANOVAs now support mixed models with between- and within-subjects factors
  (see examples at :func:`test.anova`).
* :mod:`load.fiff`: when generating epochs from raw data, a new ``tstop``
  argument allows specifying the time interval exclusive of the last sample.
* New functions:

  - :func:`table.cast_to_ndvar`
  - :func:`test.TTestInd`

* New methods:

  - :meth:`NDVar.log`
  - :meth:`NDVar.smooth`

* .. currentmodule:: eelbrain.pipeline
  :class:`MneExperiment` pipeline:

  - :meth:`MneExperiment.reset` (replacing :meth:`MneExperiment.store_state`
    and :meth:`MneExperiment.restore_state`)
  - New :attr:`MneExperiment.auto_delete_results` attribute to control whether
    invalidated results are automatically deleted.
  - :attr:`MneExperiment.screen_log_level`


New in 0.25
-----------

.. currentmodule:: eelbrain

* Installation with ``conda`` (see :doc:`installing`) and ``$ eelbrain`` launch
  script (see :doc:`getting_started`).
* API:

  - :class:`NDVar` objects now inherit names through operations.
  - Assignment to a :class:`Dataset` overwrites variable ``.name`` attributes,
    unless the :class:`Dataset` key is a pythonified version of ``.name``.

* GUI/plotting:

  - When using iPython 5 or later, GUI start and stop is now automatic. It is
    possible to revert to the old behavior with :func:`plot.configure`.
  - There are new hotkeys for most plots (see the individual plots' help for
    details).
  - Plots automatically rescale when the window is resized.

* :class:`MneExperiment`:

  - A new :attr:`MneExperiment.sessions` attribute replaces
    ``defaults['experiment']``, with support for multiple sessions in one
    experiment (see :ref:`MneExperiment-filestructure`).
  - The :attr:`MneExperiment.epochs` parameter ``sel_epoch`` has been removed,
    use ``base`` instead.
  - The setting ``raw='clm'`` has been renamed to ``raw='raw'``.
  - Custom preprocessing pipelines (see :attr:`MneExperiment.raw`).
  - The ``model`` parameter for ANOVA tests is now optional (see
    :attr:`MneExperiment.tests`).

* Reverse correlation using :func:`boosting`.
* Loading and saving ``*.wav`` files (:func:`load.wav` and :func:`save.wav`).


New in 0.24
-----------

* API:

  - :class:`MneExperiment`: For data from the NYU New York system converted
    with :mod:`mne` < 0.13, the :attr:`MneExperiment.meg_system` attribute needs
    to be set to ``"KIT-157"`` to distinguish it from data collected with the
    KIT UMD system.
  - :meth:`~testnd.ttest_rel.masked_parameter_map` method of cluster-based test
    results: use of ``pmin=None`` is deprecated, use ``pmin=1`` instead.

* New test: :class:`test.TTestRel`.
* :meth:`MneExperiment.make_report_rois` includes corrected p-values in reports
  for tests in more than one ROI    
* :meth:`MneExperiment.make_rej` now has a ``decim`` parameter to improve
  display performance.
* :class:`MneExperiment`: BEM-solution files are now created dynamically with
  :mod:`mne` and are not cached any more. This can lead to small changes
  in results due to improved numerical precision. Delete old files to free up
  space with ``mne_experiment.rm('bem-sol-file', subject='*')``.
* New :meth:`MneExperiment.make_report_coreg` method.
* New :class:`MneExperiment`: analysis parameter
  :ref:`state-connectivity`
* :class:`plot.TopoButterfly`: press ``Shift-T`` for a large topo-map with
  sensor names.


New in 0.23
-----------

* API :func:`plot.colors_for_twoway` and :func:`plot.colors_for_categorial`:
  new color model, different options.
* :class:`testnd.t_contrast_rel` contrasts can contain ``*`` to include the
  average of multiple cells.
* New :class:`NDVar` methods:  :meth:`NDVar.envelope`, :meth:`NDVar.fft`.


New in 0.22
-----------

* Epoch Rejection GUI:

  - New "Tools" menu.
  - New "Info" tool to show summary info on the rejection.
  - New "Find Bad Channels" tool to automatically find bad channels.
  - Set marked channels by clicking on topo-map.
  - Faster page redraw.

* :class:`plot.Barplot` and :class:`plot.Boxplot`: new ``cells`` argument to
  customize the order of bars/boxes.
* :class:`MneExperiment`: new method :meth:`MneExperiment.show_rej_info`.
* :class:`NDVar`: new method :meth:`NDVar.label_clusters`.
* :func:`plot.configure`: option to revert to wxPython backend for
  :mod:`plot.brain`.


New in 0.21
-----------

* :class:`MneExperiment`:

  - New epoch parameters: ``trigger_shift`` and ``vars`` (see
    :attr:`MneExperiment.epochs`).
  - :meth:`~MneExperiment.load_selected_events`: new ``vardef`` parameter to
    load variables from a test definition.
  - Log files stored in the root directory.
  - Parcellations (:attr:`MneExperiment.parcs`) based on combinations can also
    include split commands.

* New :class:`Factor` method:  :meth:`Factor.floodfill`.
* :class:`Model` methods: :meth:`~Model.get_table` replaced with
  :meth:`~Model.as_table`, new :meth:`~Model.head` and :meth:`~Model.tail`.
* API: ``.sort_idx`` methods renamed to ``.sort_index``.


New in 0.20
-----------

* :class:`MneExperiment`: new analysis parameter ``select_clusters='all'`` to
  keep all clusters in cluster tests (see
  :ref:`state-select_clusters`).
* Use :func:`testnd.configure` to limit the number of CPUs that are used in
  permutation cluster tests.

New in 0.19
-----------

* Two-stage tests (see :attr:`MneExperiment.tests`).
* Safer cache-handling. See note at :ref:`MneExperiment-intro-analysis`.
* :meth:`Dataset.head` and :meth:`Dataset.tail` methods for more efficiently
  inspecting partial Datasets.
* The default format for plots in reports is now SVG since they are displayed
  correctly in Safari 9.0. Use :func:`plot.configure` to change the default
  format.
* API: Improvements in :class:`plot.Topomap` with concomitant changes in the
  constructor signature. For examples see the `meg/topographic plotting
  <https://github.com/christianbrodbeck/Eelbrain/blob/r/0.19/examples/meg/topographic%20plotting.py>`_
  example.
* API: :class:`plot.ColorList` has a new argument called `labels`.
* API: :class:`testnd.anova` attribute :attr:`~testnd.anova.probability_maps`
  renamed to :attr:`~testnd.anova.p` analogous to other :mod:`testnd` results.
* Rejection-GUI: The option to plot the data range only has been removed.


New in 0.18
-----------

* API: The first argument for :meth:`MneExperiment.plot_annot` is now `parc`.
* API: the ``fill_in_missing`` parameter to :func:`combine` has been deprecated
  and replaced with a new parameter called ``incomplete``.
* API: Several plotting functions have a new ``xticklabels`` parameter to
  suppress x-axis tick labels (e.g. :class:`plot.UTSStat`).
* The objects returned by :mod:`plot.brain` plotting functions now contain
  a :meth:`~plot._brain_object.Brain.plot_colorbar` method to create a
  corresponding :class:`plot.ColorBar` plot.
* New function :func:`choose` to combine data in different :class:`NDVars`
  on a case by case basis.
* Rejection-GUI (:func:`gui.select_epochs`): Press Shift-i when hovering over
  an epoch to enter channels for interpolation manually.
* :meth:`MneExperiment.show_file_status` now shows the last modification date
  of each file.
* Under OS X 10.8 and newer running code under a notifier statement now
  automatically prevents the computer from going to sleep.


New in 0.17
-----------

* :attr:`MneExperiment.brain_plot_defaults` can be used to customize PySurfer
  plots in movies and reports.
* :attr:`MneExperiment.trigger_shift` can now also be a dictionary mapping
  subject name to shift value.
* The rejection GUI now allows selecting individual channels for interpolation
  using the 'i' key.
* Parcellations based on combinations of existing labels, as well as
  parcellations based on regions around points specified in MNI coordinates can
  now be defined in :attr:`MneExperiment.parcs`.
* Source space :class:`NDVar` can be indexed with lists of region names, e.g.,
  ``ndvar.sub(source=['cuneus-lh', 'lingual-lh'])``.
* API: :func:`plot.brain.bin_table` function signature changed slightly (more
  parameters, new ``hemi`` parameter inserted to match other functions' argument
  order).
* API: :func:`combine` now raises ``KeyError`` when trying to combine
  :class:`Dataset` objects with unequal keys; set ``fill_in_missing=True`` to
  reproduce previous behavior.
* API: Previously, :meth:`Var.as_factor` mapped unspecified values to
  ``str(value)``. Now they are mapped to ``''``. This also applies to
  :attr:`MneExperiment.variables` entries with unspecified values.


New in 0.16
-----------

* New function for plotting a legend for annot-files:
  :func:`plot.brain.annot_legend` (automatically used in reports).
* Epoch definitions in :attr:`MneExperiment.epochs` can now include a ``'base'``
  parameter, which will copy the given "base" epoch and modify it with the
  current definition.
* :meth:`MneExperiment.make_mov_ttest` and
  :meth:`MneExperiment.make_mov_ga_dspm` are fixed but require PySurfer 0.6.
* New function: :func:`table.melt_ndvar`.
* API: :mod:`plot.brain` function signatures changed slightly to accommodate
  more layout-related arguments.
* API: use :meth:`Brain.image` instead of :func:`plot.brain.image`.


New in 0.15
-----------

* The Eelbrain package on the PYPI is now compiled with Anaconda. This means
  that the package can now be installed into an Anaconda distribution with
  ``pip``, whereas ``easy_install`` has to be used for the Canopy distribution.
* GUI :func:`gui.select_epochs`: Set marked channels through menu (View > Mark
  Channels)
* Datasets can be saved as tables in RTF format (:meth:`Dataset.save_rtf`).
* API :class:`plot.Timeplot`: the default spread indicator changed to SEM, and
  there is a new argument for `timelabels`.
* API: :func:`test.anova` is now a function with a slightly changed signature.
  The old class has been renamed to :class:`test.ANOVA`.
* API: :func:`test.oneway` was removed. Use :func:`test.anova`.
* API: the default value of the :class:`plot.Timeplot` parameter `bottom`
  changed from `0` to `None` (determined by the data).
* API: :meth:`Factor.relabel` renamed to :meth:`Factor.update_labels`.
* Plotting: New option for the figure legend ``'draggable'`` (drag the legend
  with the mouse pointer).


New in 0.14
-----------

* API: the :class:`plot.Topomap` argument `sensors` changed to `sensorlabels`.
* GUI: The *python/Quit Eelbrain* menu command now closes all windows to ensure
  that unsaved documents are handled properly. In order to yield to the terminal
  without closing windows, use the *Go/Yield to Terminal* command
  (Command-Alt-Q).
* :class:`testnd.t_contrast_rel`:  support for unary operation `abs`.


New in 0.13
-----------

* The :func:`gui.select_epochs` GUI can now also be used to set bad channels.
  :class:`MneExperiment` subclasses will combine bad channel information from
  rejection files with bad channel information from bad channel files. Note
  that while bad channel files set bad channels for a given raw file
  globally, rejection files set bad channels only for the given epoch.
* :class:`Factor` objects can now remember a custom cell order which determines
  the order in tables and plots.
* The :meth:`Var.as_factor` method can now assign all unmentioned codes to a
  default value.
* :class:`MneExperiment`:

  - API: Subclasses should remove the ``subject`` and ``experiment`` parameters
    from :meth:`MneExperiment.label_events`.
  - API: :class:`MneExperiment` can now be imported directly from
    :mod:`eelbrain`.
  - API: The :attr:`MneExperiment._defaults` attribute should be renamed to
    :attr:`MneExperiment.defaults`.
  - A draft for a guide at :ref:`experiment-class-guide`.
  - Cached files are now saved in a separate folder at ``root/eelbrain-cache``.
    The cache can be cleared using :meth:`MneExperiment.clear_cache`. To
    preserve cached test results, move the ``root/test`` folder into the
    ``root/eelbrain-cache`` folder.


New in 0.12
-----------

* API: :class:`Dataset` construction changed, allows setting the number of
  cases in the Dataset.
* API:  :class:`plot.SensorMap2d` was renamed to :class:`plot.SensorMap`.
* :class:`~experiment.MneExperiment`:

  - API: The default number of samples for reports is now 10'000.
  - New epoch parameter ``'n_cases'``:  raise an error if an epoch definition
    does not yield expected number of trials.
  - A custom baseline period for epochs can now be specified as a parameter in
    the epoch definition (e.g., ``'baseline': (-0.2, -0.1)``). When loading
    data, specifying ``baseline=True`` uses the epoch's custom baseline.


New in 0.11
-----------

* :class:`~experiment.MneExperiment`:

  - Change in the way the covariance matrix
    is defined:  The epoch for the covariance matrix should be specified in
    ``MneExperiment.epochs['cov']``. The regularization is no longer part of
    :meth:`~experiment.MneExperiment.set_inv`, but is instead set with
    ``MneExperiment.set(cov='reg')`` or ``MneExperiment.set(cov='noreg')``.
  - New option ``cov='bestreg'`` automatically selects the regularization
    parameter for each subejct.

* :meth:`Var.as_factor` allows more efficient labeling when multiple values
  share the same label.
* API: Previously :func:`plot.configure_backend` is now :func:`plot.configure`


New in 0.10
-----------

* Tools for generating colors for categories (see :ref:`ref-plotting`).
* Plots now all largely respect matplotlib rc-parameters (see
  `Customizing Matplotlib <http://matplotlib.org/users/customizing.html>`_).
* Fixed an issue in the :mod:`testnd` module that could affect permutation based
  p-values when multiprocessing was used.


New in 0.9
----------

* :class:`Factor` API change:  The ``rep`` argument was renamed to ``repeat``.
* T-values for regression coefficients through :meth:`NDVar.ols_t`.
* :class:`~experiment.MneExperiment`: subject name patterns and eog_sns are
  now handled automatically.
* :class:`~plot.uts.UTSStat` and :class:`~plot.uv.Barplot` plots can use pooled
  error for variability estimates (on by default for related measures designs,
  can be turned off using the ``pool_error`` argument).

  - API: for consistency, the argument to specify the kind of error to plot
    changed to ``error`` in both plots.


New in 0.8
----------

* A new GUI application controls plots as well as the epoch selection GUI (see
  notes in the reference sections on :ref:`ref-plotting` and :ref:`ref-guis`).
* Randomization/Monte Carlo tests now seed the random state to make results replicable.


New in 0.6
----------

* New recipes for :ref:`recipe-regression`.


New in 0.5
----------

* The :mod:`eelbrain.lab` and :mod:`eelbrain.eellab` modules are deprecated.
  Everything can now me imported from :mod:`eelbrain` directly.


New in 0.4
----------

* Optimized ANOVA evaluation, support for unbalanced fixed effects models.
* `rpy2 <http://rpy.sourceforge.net>`_ interaction: :meth:`Dataset.from_r` to
  create a :class:`Dataset` from an R Data Frame, and :meth:`Dataset.to_r` to
  cerate an R Data Frame from a :class:`Dataset`.


New in 0.3
----------

* Optimized clustering for cluster permutation tests.


New in 0.2
----------

* :class:`gui.SelectEpochs` Epoch rejection GIU has a new "GA" button to plot
  the grand average of all accepted trials
* Cluster permutation tests in :mod:`testnd` use multiple cores; To disable
  multiprocessing set ``eelbrain._stats.testnd.multiprocessing = False``.


New in 0.1.7
------------

* :class:`gui.SelectEpochs` can now be initialized with a single
  :class:`mne.Epochs` instance (data needs to be preloaded).
* Parameters that take :class:`NDVar` objects now also accept
  :class:`mne.Epochs` and :class:`mne.fiff.Evoked` objects.


New in 0.1.5
------------

* :py:class:`plot.topo.TopoButterfly` plot: new keyboard commands (``t``,
  ``left arrow``, ``right arrow``).


.. _Proloy Das: https://ece.umd.edu/~proloy/
