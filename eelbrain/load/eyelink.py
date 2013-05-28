'''
Created on Jun 9, 2012

@author: christian
'''
import fnmatch
import os
import re
import shutil
from subprocess import Popen
import tempfile

import numpy as np

from .. import ui
from ..utils.subp import get_bin
from ..vessels.data import dataset, datalist, var

__all__ = ['Edf', 'read_edf', 'read_edf_events', 'read_edf_samples']



class Edf(object):
    """
    Class for reading an eyelink .edf file.

    Reads an eyelink .edf file and extracts epoch acceptability
    based on contamination with ocular artifacts (saccades and blinks).
    An edf file reader is initialized with the path to the corresponding
    file::

        >>> path = '/path/to/edf.edf'
        >>> edf = load.eyelink.Edf(path)

    There are several ways of retrieving trial acceptability values (see the
    used methods' documentation for more information):


    1. **For all triggers:**
       Acceptability for all triggers can be added to a dataset with a single
       command if the dataset contains the same events as the edf file::

          >>> edf.mark_all(ds, ...)


    2. **For a subset of triggers:**
       Often it is more efficient to compute acceptability only for a subset of
       the triggers contained in the edf file. For those cases, the trigger time
       should first be added to the complete dataset with::

           >>> edf.add_T_to(ds)

       Now, the dataset can be decimated::

           >>> ds = ds.subset(...)

       and acceptability can be added for the subset::

           >>> edf.mark(ds, ...)

    3. **Customized:**
       For a more customized use, all triggers from the edf can be retrieved using::

           >>> ds_edf = edf.get_triggers()

       The ``ds_edf['t_edf']`` time variable can be used to add trigger time values to
       arbitrary events, which can then in turn be used with::

           >>> edf.mark(...)

    """
    def __init__(self, path=None, samples=False):
        """
        Parameters
        ----------
        path : str(path) | None
            Path to the .edf file. The If path contains '*', the files matching
            the pattern are concatenated. If None, a file-open dialogue will be
            displayed.

        """
        if path is None:
            path = ui.ask_file("Load an eyelink .edf file", "Pick the edf file",
                               ext=[('edf', 'eyelink data format')])

        # find all paths from which to read
        self.path = path
        if '*' in path:
            head, tail = os.path.split(path)
            if '*' in head:
                err = ("Invalid path: %r. All edf files need to be in the same "
                       "directory." % path)
                raise ValueError(err)

            fnames = sorted(fnmatch.filter(os.listdir(head), tail))
            self.paths = [os.path.join(head, fname) for fname in fnames]
        else:
            self.paths = [path]

        triggers = []
        artifacts = []
        pos = []
        for path in self.paths:
            edf_asc = read_edf(path)
            triggers += find_edf_triggers(edf_asc)
            artifacts += find_edf_artifacts(edf_asc)
            if samples:
                pos += find_edf_pos(edf_asc)

        dtype = [('T', np.uint32), ('Id', np.uint8)]
        self.triggers = np.array(triggers, dtype=dtype)

        dtype = np.dtype([('event', np.str_, 6), ('start', np.uint32), ('stop', np.uint32)])
        self.artifacts = np.array(artifacts, dtype=dtype)

        if samples:
            self.time = np.array([item[0] for item in pos], dtype=np.uint32)
            self.xpos = np.array([item[1] for item in pos], dtype=np.float16)
            self.ypos = np.array([item[2] for item in pos], dtype=np.float16)
            self.pdia = np.array([item[3] for item in pos], dtype=np.float16)

    def __repr__(self):
        return "Edf(%r)" % self.path

    def assert_Id_match(self, ds=None, Id='eventId'):
        """
        Make sure the Edf and another event list describe the same events.

        Raises an error if the Ids in ``Id`` do not match the Ids in the Edf
        file(s).

        Parameters
        ----------
        ds : None | dataset
            Dataset with events.
        Id : str | array
            If `ds` is a dataset, `Id` should be a string naming the variable
            in `ds` containing the event IDs. If `ds` is None, `Id` should be
            a series of event Ids.

        """
        if isinstance(Id, str):
            Id = ds[Id]

        ID_edf = self.triggers['Id']
        if len(Id) != len(ID_edf):
            lens = (len(Id), len(ID_edf))
            mm = min(lens)
            for i in xrange(mm):
                if Id[i] != ID_edf[i]:
                    mm = i
                    break

            args = (getattr(ds, 'name', 'None'), self.path) + lens + (mm,)
            err = ("dataset %r containes different number of events from edf "
                   "file %r (%i vs %i); first mismatch at %i." % args)
            raise ValueError(err)

        check = (Id == ID_edf)
        if not all(check):
            err = "Event ID mismatch: %s" % np.where(check == False)[0]
            raise ValueError(err)

    def add_T_to(self, ds, Id='eventID', t_edf='t_edf'):
        """
        Add edf trigger times as a variable to dataset ds.
        These can then be used for Edf.add_by_T(ds) after ds hads been
        decimated.

        Parameters
        ----------
        ds : dataset
            The dataset to which the variable is added
        Id : str | var | None
            variable (or its name in the dataset) containing event IDs. Values
            in this variable are checked against the events in the EDF file,
            and an error is raised if there is a mismatch. This test can be
            skipped by setting Id=None.
        t_edf : str
            Name for the target variable holding the edf trigger times.

        """
        if Id:
            self.assert_Id_match(ds=ds, Id=Id)
            if isinstance(Id, str):
                Id = ds[Id]

        ds[t_edf] = var(self.triggers['T'])

    def filter(self, ds, tstart= -0.1, tstop=0.6, use=['ESACC', 'EBLINK'],
               T='t_edf'):
        """
        Return a copy of the dataset ``ds`` with all bad events removed. A
        dataset containing all the bad events is stored in
        ``ds.info['rejected']``.

        Parameters
        ----------
        ds : dataset
            The dataset that is to be filtered
        tstart : scalar
            Start of the time window in which to look for artifacts
        tstop : scalar
            End of the time window in which to look for artifacts
        use : list of str
            List of events types which are to be treated as artifacts (possible
            are 'ESACC' and 'EBLINK')
        T : str | var
            Variable describing edf-relative timing for the events in ``ds``.
            Usually this is a string key for a variable in ``ds``.

        """
        if isinstance(T, str):
            T = ds[T]
        accept = self.get_accept(T, tstart=tstart, tstop=tstop, use=use)
        accepted = ds.subset(accept)
        rejected = ds.subset(accept == False)
        accepted.info['rejected'] = rejected
        return accepted

    def get_accept(self, T=None, tstart= -0.1, tstop=0.6, use=['ESACC', 'EBLINK']):
        """
        returns a boolean var indicating for each epoch whether it should be
        accepted or not based on ocular artifacts in the edf file.

        Parameters
        ----------
        T : array-like | None
            List of time points (in the edf file's time coordinates). If None,
            the edf's trigger events are used.
        tstart : scalar
            start of the epoch relative to the event (in seconds)
        tstop : scalar
            end of the epoch relative to the even (in seconds)

        """
        if T is None:
            T = self.triggers['T']

        # conert to ms
        start = int(tstart * 1000)
        stop = int(tstop * 1000)

        self._debug = []

        # get data for triggers
        N = len(T)
        accept = np.empty(N, np.bool_)

        X = tuple(self.artifacts['event'] == name for name in use)
        idx = np.any(X, axis=0)
        artifacts = self.artifacts[idx]

        for i, t in enumerate(T):
            starts_before_tstop = artifacts['start'] < t + stop
            stops_after_tstart = artifacts['stop'] > t + start
            overlap = np.all((starts_before_tstop, stops_after_tstart), axis=0)
            accept[i] = not np.any(overlap)

            self._debug.append(overlap)

        return accept

    def get_T(self, name='t_edf'):
        "returns all trigger times in the dataset"
        return var(self.triggers['T'], name=name)

    def get_triggers(self, Id='Id', T='t_edf'):
        """
        Returns a dataset with trigger Ids and corresponding Edf time values

        """
        ds = dataset()
        ds[Id] = var(self.triggers['Id'])
        ds[T] = self.get_T(name=T)
        return ds

    def mark(self, ds, tstart= -0.1, tstop=0.6, good=None, bad=False,
             use=['ESACC', 'EBLINK'], T='t_edf', target='accept'):
        """
        Mark events in ds as acceptable or not. ds needs to contain edf trigger
        times in a variable whose name is specified by the ``T`` argument.

        Parameters
        ----------
        ds : dataset
            dataset that contains the data to work with.
        tstart : scalar
            start of the time window relevant for rejection.
        tstop : scalar
            stop of the time window relevant for rejection.
        good : bool | None
            vale assigned to epochs that should be retained based on
            the eye-tracker data.
        bad : bool | None
            value that is assigned to epochs that should be rejected
            based on the eye-tracker data.
        use : list of str
            Artifact categories to include
        T : var
            variable providing the trigger time values
        target : var
            variable to which the good/bad values are assigned (if it does not
            exist, a new variable will be created with all values True
            initially)

        """
        if isinstance(target, str):
            if target in ds:
                target = ds[target]
            else:
                ds[target] = target = var(np.ones(ds.n_cases, dtype=bool))

        if isinstance(T, str):
            T = ds[T]

        accept = self.get_accept(T, tstart=tstart, tstop=tstop, use=use)
        if good is not None:
            target[accept] = good
        if bad is not None:
            target[np.invert(accept)] = bad

    def mark_all(self, ds, tstart= -0.1, tstop=0.6, good=None, bad=False,
                 use=['ESACC', 'EBLINK'],
                 Id='eventID', target='accept'):
        """
        Mark each epoch in the ds for acceptability based on overlap with
        blinks and saccades. ds needs to contain the same number of triggers
        as the edf file. For adding acceptability to a decimated ds, use
        Edf.add_T_to(ds, ...) and then Edf.mark(ds, ...).

        Parameters
        ----------
        ds : dataset
            dataset that contains the data to work with.
        tstart : scalar
            start of the time window relevant for rejection.
        tstop : scalar
            stop of the time window relevant for rejection.
        good :
            vale assigned to epochs that should be retained based on
            the eye-tracker data.
        bad :
            value that is assigned to epochs that should be rejected
            based on the eye-tracker data.
        use : list of str
            Artifact categories to include
        Id : var | None
            variable containing trigger Ids for asserting that the dataset
            contains the same triggers as rhe edf file(s). The test can be
            skipped by setting Id=None
        target : var
            variable to which the good/bad values are assigned (if it does not
            exist, a new variable will be created with all values True
            initially)

        """
        if Id:
            self.assert_Id_match(ds=ds, Id=Id)
            if isinstance(Id, str):
                Id = ds[Id]

        T = self.get_T()
        self.mark(ds, tstart=tstart, tstop=tstop, good=good, bad=bad, use=use,
                  T=T, target=target)


def events(fname, samples=False):
    """Read events from an edf file

    Parameters
    ----------
    fname : str
        Filename.
    samples : bool
        Read samples as well as events.
    """
    edf = Edf(fname, samples=samples)
    ds = dataset(info={'edf': edf})
    ds['t_edf'] = var(edf.triggers['T'])
    ds['eventID'] = var(edf.triggers['Id'])
    return ds


def add_artifact_epochs(ds, tmin= -0.2, tmax=0.6, crop=True, esacc='sacc',
                        eblink='blink', t_edf='t_edf'):
    """Add a datalist containing artifact information by event

    Parameters
    ----------
    ds : dataset
        Dataset with 'edf' entry in its info dictionary (usually a dataset
        returned by ``load.eyelink.events()``)
    tmin, tmax : scalar
        Relative start and end points of the epoch (in seconds).
    crop : bool
        Crop events to epoch beginning and end (i.e., if an artifact starts
        before the epoch, set its start to the first sample in the epoch).
    esacc, eblink : None | str
        Name for the variable containing the corresponding information. If
        None, the corresponding variable is not added.
    t_edf : str
        Name of the ds variable containing edf times.

    Returns
    -------
    ds : dataset
        Returns the input dataset for consistency with similar functions; the
        dataset is modified in place.
    """
    edf = ds.info['edf']
    start = edf.artifacts['start']
    stop = edf.artifacts['stop']
    is_blink = (edf.artifacts['event'] == 'EBLINK')
    is_sacc = (edf.artifacts['event'] == 'ESACC')

    # edf times are in ms; convert them to s:
    dtype = np.dtype([('event', np.str_, 6), ('start', np.float64), ('stop', np.float64)])
    artifacts_s = edf.artifacts.astype(dtype)
    artifacts_s['start'] /= 1000.
    artifacts_s['stop'] /= 1000.

    sacc = datalist(name=esacc)
    blink = datalist(name=eblink)
    for t in ds[t_edf]:
        t_s = t / 1000.
        epoch_idx = np.logical_and(stop > t + tmin, start < t + tmax)
        if esacc:
            idx = np.logical_and(epoch_idx, is_sacc)
            epoch = artifacts_s[idx]
            epoch['start'] -= t_s
            epoch['stop'] -= t_s
            if crop and len(epoch):
                if epoch['start'][0] < tmin:
                    epoch['start'][0] = tmin
                if epoch['stop'][-1] > tmax:
                    epoch['stop'][-1] = tmax
            sacc.append(epoch)
        if eblink:
            idx = np.logical_and(epoch_idx, is_blink)
            epoch = artifacts_s[idx]
            epoch['start'] -= t_s
            epoch['stop'] -= t_s
            if crop and len(epoch):
                if epoch['start'][0] < tmin:
                    epoch['start'][0] = tmin
                if epoch['stop'][-1] > tmax:
                    epoch['stop'][-1] = tmax
            blink.append(epoch)

    if esacc:
        ds.add(sacc)
    if eblink:
        ds.add(blink)
    return ds


def read_edf(fname, what='events'):
    """
    Read the content of an edf file as text using edf2asc

    Convert an "eyelink data format" (.edf) file to a temporary directory
    and parse its content.

    Parameters
    ----------
    fname : str
        Filename.
    what : 'all' | 'events' | 'samples'
        What type of information to read
    """
    if not os.path.exists(fname):
        err = "File does not exist: %r" % fname
        raise ValueError(err)

    temp_dir = tempfile.mkdtemp()

    # edf2asc does not seem to handle spaces in filenames?
    if ' ' in fname:
        dst = os.path.join(temp_dir, os.path.basename(fname).replace(' ', '_'))
#         shutil.copy(fname, dst)
        os.symlink(fname, dst)
        fname = dst

    # construct the conversion command
    cmd = [get_bin('edfapi', 'edf2asc'),  # options in Manual p. 106
           '-t', ]  # use only tabs as delimiters
    if what == 'events':
        cmd.append('-e')  # outputs event data only
    elif what == 'samples':
        cmd.append('-s')  # outputs sample data only
    elif what == 'all':
        raise NotImplementedError()
    else:
        raise ValueError("what must be 'events' or 'samples', not %r" % what)

    cmd.extend(('-nst',  # blocks output of start events
                '-p', temp_dir,  # writes output with same name to <path> directory
                fname))

    sp = Popen(cmd)
    sp.communicate()
#     out = check_output(cmd)

    # find asc file
    name, _ = os.path.splitext(os.path.basename(fname))
    ascname = os.path.extsep.join((name, 'asc'))
    asc_path = os.path.join(temp_dir, ascname)
    with open(asc_path) as asc_file:
        asc_str = asc_file.read()

    # clean
    shutil.rmtree(temp_dir)
    return asc_str


def find_edf_triggers(asc_str):
    """Find artifacts in an edf asci representation

    Parameters
    ----------
    asc_str : str
        String with edf asci represenation as returned by read_edf.
    """
    re_trigger = re.compile(r'\bMSG\t(\d+)\tMEG Trigger: (\d+)')
    triggers = re_trigger.findall(asc_str)
    return triggers


def find_edf_artifacts(asc_str, kind='EBLINK|ESACC'):
    """Find artifacts in an edf asci representation

    Parameters
    ----------
    asc_str : str
        String with edf asci represenation as returned by read_edf.
    kind : 'EBLINK|ESACC' | 'EBLINK' | 'ESACC'
        Kind of artifact to search.
    """
    for kind_part in kind.split('|'):
        if kind_part not in ['EBLINK', 'ESACC']:
            raise ValueError("invalid kind parameter: %r" % kind)
    re_artifact = re.compile(r'\b(%s)\t[LR]\t(\d+)\t(\d+)' % kind)
    artifacts = re_artifact.findall(asc_str)
    return artifacts


def find_edf_pos(asc_str):
    """Find position values in an edf asci representation

    Parameters
    ----------
    asc_str : str
        String with edf asci represenation as returned by read_edf.
    """
    re_pos = re.compile(r'\b(\d+)\t(\d+\.\d*)\t(\d+\.\d*)\t(\d+\.\d*)')
    pos = re_pos.findall(asc_str)
    return pos
