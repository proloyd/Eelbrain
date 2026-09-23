import json
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np

from eelbrain import Pipeline
from eelbrain.pipeline import PrimaryEpoch, RawICA, RawMaxwell
from eelbrain._experiment.pathing import ica_file_path, rej_file_path, test_basename as result_basename
from eelbrain._experiment.preprocessing import ica_input_name, raw_input_name


def _write_dataset_description(root: Path, name: str) -> None:
    (root / 'dataset_description.json').write_text(json.dumps({'Name': name, 'BIDSVersion': '1.10.0'}))


def _make_raw(path, triggers: tuple[int, ...]) -> None:
    info = mne.create_info(['MEG 001', 'STI 014'], 100., ['mag', 'stim'])
    data = np.zeros((2, 200))
    for index, trigger in enumerate(triggers, 1):
        data[1, index * 20] = trigger
    raw = mne.io.RawArray(data, info, verbose='error')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(path, overwrite=True, verbose='error')


def test_acquisition_derivative_paths():
    state = {
        'subject': '01',
        'session': '',
        'task': 'test',
        'acquisition': 'highres',
        'run': '1',
        'raw': 'ica',
        'epoch': 'test',
        'epoch_rejection': 'manual',
    }
    assert ica_file_path(state, 'ica', datatype='meg') == Path('derivatives/mne/sub-01/meg/sub-01_acq-highres_run-1_desc-ica_ica.fif')
    assert rej_file_path(state, datatype='meg') == Path('derivatives/mne/sub-01/meg/sub-01_acq-highres_run-1_raw-ica_epoch-test_rej-manual_epoch.pickle')
    assert result_basename(state, datatype='meg') == 'acq-highres_run-1_meg'


def test_acquisitions_are_independent_analysis_branches(tmp_path):
    _write_dataset_description(tmp_path, 'acquisition-test')
    meg_dir = tmp_path / 'sub-01' / 'meg'
    for acquisition, triggers in {'a': (1,), 'b': (1, 2)}.items():
        for run in ('1', '2'):
            path = meg_dir / f'sub-01_task-test_acq-{acquisition}_run-{run}_meg.fif'
            _make_raw(path, triggers)

    class AcquisitionExperiment(Pipeline):
        stim_channel = 'STI 014'
        epochs = {
            'test': PrimaryEpoch('test', tmin=0, tmax=0.01, baseline=False),
        }

    experiment = AcquisitionExperiment(tmp_path)
    assert experiment.get_field_values('acquisition') == ['a', 'b']
    assert experiment._recordings == {
        ('01', '', 'test', acquisition, run)
        for acquisition in ('a', 'b')
        for run in ('1', '2')
    }
    assert experiment._runs_for == {
        ('01', '', 'test', acquisition): ['1', '2']
        for acquisition in ('a', 'b')
    }

    event_paths = {}
    for acquisition, n_events in {'a': 2, 'b': 4}.items():
        experiment.set(epoch='test', epoch_rejection='', acquisition=acquisition)
        events = experiment.load_selected_events()
        assert events.n_cases == n_events

        experiment.set(run='1')
        raw_ctx = experiment._resolve_derivative(raw_input_name('raw'), options={'noise': False})
        assert raw_ctx.node.path(raw_ctx).name == f'sub-01_task-test_acq-{acquisition}_run-1_meg.fif'
        event_paths[acquisition] = experiment._resolve_derivative('events').artifact_path

    assert event_paths['a'] != event_paths['b']


def test_ignore_runs_excludes_ica_and_head_position_recordings(tmp_path):
    """ignore_runs must also be honored by non-epoch derivatives that
    concatenate across runs, not just by epoch/event aggregation.

    Regression coverage for the same Pipeline.__init__ bug as
    test_epoch_run's ignore_runs check: Pipeline._recordings (built from
    find_matching_paths) used to include every run regardless of
    ignore_runs, so ICAInput (which concatenates all of a subject's runs
    for the ICA fit once the ICA step follows a RawMaxwell step, see
    RawICA._concatenate_runs) and CanonicalHeadPositionDerivative would
    have silently used the ignored run's recording too.

    Only entity discovery (Pipeline.__init__: find_matching_paths,
    _recordings, _runs_for) and the resulting node wiring are under test
    here, not any actual raw data; nothing in this test reads a raw file's
    content, so the source files only need BIDS-valid names, not
    MNE-readable content (unlike test_acquisitions_are_independent_analysis_branches
    above, which does load real event data and needs _make_raw).
    """
    _write_dataset_description(tmp_path, 'ignore-runs-test')
    meg_dir = tmp_path / 'sub-01' / 'meg'
    meg_dir.mkdir(parents=True)
    for run in ('1', '2', '3'):
        (meg_dir / f'sub-01_task-test_run-{run}_meg.fif').touch()

    class IgnoreRunExperiment(Pipeline):
        stim_channel = 'STI 014'
        ignore_entities = {'ignore_runs': ('2',)}
        raw = {
            'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=.9, st_only=True, st_overlap=False),
            'ica': RawICA('tsss', method='fastica', n_components=1),
        }

    experiment = IgnoreRunExperiment(tmp_path)
    assert experiment._runs == ('1', '3')
    assert experiment._recordings == {('01', '', 'test', '', run) for run in ('1', '3')}

    ica_node = experiment._derivatives._nodes[ica_input_name('ica')]
    assert ica_node.pipe._concatenate_runs is True
    ctx = SimpleNamespace(state={'subject': '01', 'session': '', 'acquisition': ''})
    states = ica_node._source_states(ctx, ['test'])
    assert {state['run'] for state in states} == {'1', '3'}
