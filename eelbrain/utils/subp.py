'''

for permission errors: try ``os.chmod`` or ``os.chown``


subprocess documentation
------------------------

http://docs.python.org/library/subprocess.html
http://www.doughellmann.com/PyMOTW/subprocess/
http://explanatorygap.net/2010/05/10/python-subprocess-over-shell/
http://codeghar.wordpress.com/2011/12/09/introduction-to-python-subprocess-module/


Created on Mar 4, 2012

@author: christian
'''

import cPickle as pickle
import fnmatch
import logging
import os
import re
import shutil
import subprocess
import tempfile

from eelbrain import ui
from eelbrain.load.brainvision import vhdr as _vhdr


__hide__ = ['os', 'shutil', 'subprocess', 'tempfile', 're', 'fnmatch', 'pickle',
            'np',
            'ui']
#__all__ = [
##           'forward',
#           'kit2fiff', 
#           'process_raw', 
#           'set_bin_dirs',
#           'mne_experiment'
#           ] 

def _set_bin_dirs(mne=None, freesurfer=None, edfapi=None):
    "Setup for binary packages"
    if mne:
        mne_bin = mne
        mne_root, _ = os.path.split(mne)
        os.environ['MNE_ROOT'] = mne_root
        os.environ['DYLD_LIBRARY_PATH'] = os.path.join(mne_root, 'lib')

        if 'PATH' in os.environ:
            os.environ['PATH'] += ':%s' % mne_bin
        else:
            os.environ['PATH'] = mne_bin

    if freesurfer:
        fs_home, _ = os.path.split(freesurfer)
        os.environ['FREESURFER_HOME'] = fs_home



# keep track of whether the mne dir has been successfully set
_cfg_path = os.path.join(os.path.dirname(__file__), 'bin_cfg.pickled')
_bin_dirs = {'mne': 'None',
             'freesurfer': 'None',
             'edfapi': 'None'}

try:
    _bin_dirs.update(pickle.load(open(_cfg_path)))
    _set_bin_dirs(**_bin_dirs)
except:
    logging.info("subp: loading paths failed at %r" % _cfg_path)

def get_bin(package, name):
    if package not in _bin_dirs:
        raise KeyError("Unknown binary package: %r" % package)

    bin_path = os.path.join(_bin_dirs[package], name)
    if not os.path.exists(bin_path):
        set_bin_dir(package)
        bin_path = os.path.join(_bin_dirs[package], name)

    return bin_path


def _is_bin_path(package, path):
    if package == 'mne':
        test = ['mne_add_patch_info', 'mne_analyze']
    elif package == 'freesurfer':
        test = ['mri_annotation2label']
    elif package == 'edfapi':
        test = ['edf2asc']
    else:
        raise KeyError("Unknown package: %r" % package)

    test_paths = [os.path.join(path, testname) for testname in test]
    if all(map(os.path.exists, test_paths)):
        return True
    else:
        msg = ("You need to select a directory containing the following "
               "executables: %r" % test)
        ui.message("Invalid Binary Directory", msg, 'error')


def set_bin_dir(package):
    """
    Change the location from which binaries are used.

    package : str
        Binary package for which to set the directory. One from:
        ``['mne', 'freesurfer', 'edfapi']``

    """
    have_valid_path = False
    while not have_valid_path:
        title = "Select %r bin Directory" % package
        message = ("Please select the directory containing the binaries for "
                   "the %r package." % package)
        answer = ui.ask_dir(title, message, must_exist=True)
        if answer:
            if _is_bin_path(package, answer):
                _bin_dirs[package] = answer
                pickle.dump(_bin_dirs, open(_cfg_path, 'w'))
                _set_bin_dirs(**{package: answer})
                have_valid_path = True
        else:
            raise IOError("%r bin directory not set" % package)





# create dictionary of available sns files
_sns_files = {}
_sns_dir = __file__
for i in xrange(2):
    _sns_dir = os.path.dirname(_sns_dir)
_sns_dir = os.path.join(_sns_dir, 'Resources', 'sns')
for name in ['NYU-nellab']:
    _sns_files[name] = os.path.join(_sns_dir, name + '.txt')


_verbose = 1


class edf_file:
    """
    Converts an "eyelink data format" (.edf) file to a temporary directory
    and parses its content.
    
    """
    def __init__(self, path):
        # convert
        if not os.path.exists(path):
            err = "File does not exist: %r" % path
            raise ValueError(err)

        self.source_path = path
        self.temp_dir = tempfile.mkdtemp()
        cmd = [get_bin('edfapi', 'edf2asc'), # options in Manual p. 106
               '-t', # use only tabs as delimiters
               '-e', # outputs event data only
               '-nse', # blocks output of start events
               '-p', self.temp_dir, # writes output with same name to <path> directory
               path]

        _run(cmd)

        # find asc file
        name, _ = os.path.splitext(os.path.basename(path))
        ascname = os.path.extsep.join((name, 'asc'))
        self.asc_path = os.path.join(self.temp_dir, ascname)
        self.asc_file = open(self.asc_path)
        self.asc_str = self.asc_file.read()

        # find trigger events
        #                           MSG   time    msg...      ID  
        re_trigger = re.compile(r'\bMSG\t(\d+)\tMEG Trigger: (\d+)')
        self.triggers = re_trigger.findall(self.asc_str)

        # find artifacts
        #                            type                    start   end
        re_artifact = re.compile(r'\b(ESACC|EBLINK)\t[LR]\t(\d+)\t(\d+)')
        self.artifacts = re_artifact.findall(self.asc_str)

    def __del__(self):
        shutil.rmtree(self.temp_dir)

    def __repr__(self):
        return 'edf_file(%r)' % self.source_path



class marker_avg_file:
    def __init__(self, path):
        # Parse marker file, based on Tal's pipeline:
        regexp = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
        output_lines = []
        for line in open(path):
            match = regexp.search(line)
            if match:
                output_lines.append('\t'.join(match.groups()))
        txt = '\n'.join(output_lines)

        fd, self.path = tempfile.mkstemp(suffix='hpi', text=True)
        f = os.fdopen(fd, 'w')
        f.write(txt)
        f.close()

    def __del__(self):
        os.remove(self.path)



def _format_path(path, fmt, is_new=False):
    "helper function to format the path to mne files"
    if not isinstance(path, basestring):
        path = os.path.join(*path)

    if fmt:
        path = path.format(**fmt)

    # test the path
    path = os.path.expanduser(path)
    if is_new or os.path.exists(path):
        return path
    else:
        raise IOError("%r does not exist" % path)



def kit2fiff(paths=dict(mrk=None,
                        elp=None,
                        hsp=None,
                        rawtxt=None,
                        rawfif=None),
             sns='NYU-nellab',
             sfreq=1000, lowpass=100, highpass=0,
             stim=xrange(168, 160, -1), stimthresh=2.5, add=None, #(188, 190), #xrange()
             aligntol=25, overwrite=False):
    """
    Calls the ``mne_kit2fiff`` binary which reads multiple input files and 
    combines them into a fiff file. Implemented after Gwyneth's Manual; for 
    more information see the mne manual (p. 222). 
    
    **Arguments:**
    
    paths : dict
        Dictionary containing paths to input and output files. 
        Needs the folowing keys:
        'mrk', 'elp', 'hsp', 'sns', 'rawtxt', 'outfif' 
        
    experiment : str
        The experiment name as it appears in file names.
    
    highpass : scalar
        The highpass filter corner frequency (only for file info, does not
        filter the data). 0 Hz for DC recording.
    
    lowpass : scalar
        like highpass
    
    meg_sdir : str or tuple (see above)
        Path to the subjects's meg directory. If ``None``, a file dialog is 
        displayed to ask for the directory.
    
    sfreq : scalar
        samplingrate of the data
    
    stimthresh : scalar
        The threshold value used when synthesizing the digital trigger channel
    
    add : sequence of int | None
        channels to include in addition to the 157 default MEG channels and the
        digital trigger channel. These numbers refer to the scanning order 
        channels as listed in the sns file, starting from one.
    
    stim : iterable over ints
        trigger channels that are used to reconstruct the event cue value from 
        the separate trigger channels. The default ``xrange(168, 160, -1)`` 
        reconstructs the values sent through psychtoolbox.
        
    aligntol : scalar
        Alignment tolerance for coregistration
        
    overwrite : bool
        Automatically overwrite the target fiff file if it already exists.
    
    """
    # get all the paths
    mrk_path = paths.get('mrk')
    elp_file = paths.get('elp')
    hsp_file = paths.get('hsp')
    raw_file = paths.get('rawtxt')
    out_file = paths.get('rawfif')

    if sns in _sns_files:
        sns_file = _sns_files[sns]
    elif os.path.exists(sns):
        sns_file = sns
    else:
        err = ("sns needs to the be name of a provided sns file (%s) ro a valid"
               "path to an sns file" % ', '.join(map(repr, _sns_files)))
        raise IOError(err)

    # convert the marker file
    mrk_file = marker_avg_file(mrk_path)
    hpi_file = mrk_file.path

    # make sure target path exists
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    elif not overwrite and os.path.exists(out_file):
        if not ui.ask("Overwrite?", "Target File Already Exists at: %r. Should "
                      "it be replaced?" % out_file):
            return

    cmd = [get_bin('mne', 'mne_kit2fiff'),
           '--elp', elp_file,
           '--hsp', hsp_file,
           '--sns', sns_file,
           '--hpi', hpi_file,
           '--raw', raw_file,
           '--out', out_file,
           '--stim', ':'.join(map(str, stim)), # '161:162:163:164:165:166:167:168'
           '--sfreq', sfreq,
           '--aligntol', aligntol,
           '--lowpass', lowpass,
           '--highpass', highpass,
           '--stimthresh', stimthresh,
           ]

    if add:
        cmd.extend(('--add', ':'.join(map(str, add))))

    _run(cmd)

    # TODO: rename additional channels
    # how do I know their names?? ~/Desktop/test/EOG  has MISC 28 and MISC 30
#    if add:
#        cmd = [os.path.join(mne_dir, 'bin', 'mne_rename_channels'),
#               '--fif', out_file,
#               ]

    if not os.path.exists(out_file):
        raise RuntimeError("kit2fiff failed (see above)")


def brain_vision2fiff(vhdr_file=None, dig=None, orignames=False, eximia=False):
    """
    Convert a set of brain vision files to a fiff file.
    
    vhdr_file : str(path) | None
        Path to the header file; if None, a file dialog will ask for a file.
    
    
    **mne_brain_vision2fiff arguments**
    
    dig : str(path) | None
        The digitization data file.
    orignames : bool
        Keep original electrode labels.
    eximia : bool
        These are Nexstim eXimia data. Interpret the first four channels 
        in a special way.
    
    """
    vhdr = _vhdr(vhdr_file)
    tempdir = tempfile.mkdtemp()

    # vhdr
    new_vhdr_file = os.path.join(tempdir, vhdr.basename)
    os.symlink(vhdr.path, new_vhdr_file)

    # data
    new_data_file = os.path.join(tempdir, os.path.basename(vhdr.datafile))
    os.symlink(vhdr.datafile, new_data_file)

    # vmrk
    vmrk_file = vhdr.markerfile
    new_vmrk_file = os.path.join(tempdir, os.path.basename(vmrk_file))
    prepare_vmrk(vmrk_file, new_vmrk_file)

    # destination
    vhdr_root, _ = os.path.splitext(vhdr.path)
    out_file = os.extsep.join((vhdr_root + '_raw', 'fif'))

    cmd = [get_bin('mne', 'mne_brain_vision2fiff'),
           '--header', new_vhdr_file,
           '--out', out_file]
    if dig:
        cmd.extend(['--dig', dig])
    if orignames:
        cmd.append('--orignames')
    if eximia:
        cmd.append('--eximia')

    out, err = _run(cmd, cwd=tempdir)
    shutil.rmtree(tempdir)
    if not os.path.exists(out_file):
        print out
        print err
        raise RuntimeError("brain_vision2fiff failed (see above)")



def prepare_vmrk(vmrk, dest):
    """
    Prepares events in a brain vision .vmrk file for mne_brain_vision2fiff by
    setting all event types to "Stimulus".
    
    vmrk : str(path)
        path to the original vmrk file
    
    dest : str(path)
        detination for the modified vmrk file
    
    """
    # Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,
    # <Size in data points>, <Channel number (0 = marker is related to all channels)>
    rdr = re.compile('Mk([0-9]+)=(\w+),(\w+),(\w+),(\w+),(\w+)')
    temp = 'Mk%s=%s,%s,%s,%s,%s' + os.linesep
    OUT = []
    for line in open(vmrk):
        m = rdr.match(line)
        if m:
            Id, stype, desc, pos, size, chan = m.groups()
            stype = 'Stimulus'
            desc = 'S%s' % desc[1:]
            OUT.append(temp % (Id, stype, desc, pos, size, chan))
        else:
            OUT.append(line)

    open(dest, 'w').writelines(OUT)



def process_raw(raw, save='{raw}_filt', args=['projoff'], rm_eve=True, **kwargs):
    """
    Calls ``mne_process_raw`` to process raw fiff files. All 
    options can be submitted in ``args`` or as kwargs (for a description of
    the options, see mne manual 2.7.3, ch. 4 / p. 41)
    
    raw : str(path)
        Source file. 
    save : str(path)
        Destination file. ``'{raw}'`` will be replaced with the raw file path
        without the extension.
    args : list of str
        Options that do not require a value (e.g., to specify ``--filteroff``,
        use ``args=['filteroff']``).
    rm_eve : bool
        Remove the event file automatically generated by ``mne_process_raw``
    **kwargs:** 
        ``mne_process_raw`` options that require values (see example).
    
    Example::
    
        >>> process_raw('/some/file_raw.fif', highpass=8, lowpass=12)
    
    """
    raw_name, _ = os.path.splitext(raw)
    eve_file = '%s-eve.fif' % raw_name
    if raw_name.endswith('_raw'):
        raw_name = raw_name[:-4]

    save = save.format(raw=raw_name)

    cmd = [get_bin('mne', 'mne_process_raw'),
           '--raw', raw,
           '--save', save]

    for arg in args:
        cmd.append('--%s' % arg)

    for key, value in kwargs.iteritems():
        cmd.append('--%s' % key)
        cmd.append(value)

    _run(cmd)
    if rm_eve:
        try:
            os.remove(eve_file)
        except Exception as err:
            print "Could not remove %r" % eve_file
            print err



def _run(cmd, v=None, cwd=None):
    """
    cmd: list of strings
        command that is submitted to subprocess.Popen.
    v : 0 | 1 | 2 | None
        verbosity level (0: nothing;  1: stderr;  2: stdout;  None: use 
        _verbose module attribute)
    
    """
    if v is None:
        v = _verbose

    if v > 1:
        print "> COMMAND:"
        for line in cmd:
            print repr(line)

    cmd = [unicode(c) for c in cmd]
    sp = subprocess.Popen(cmd, cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()

    if v > 1:
        print "\n> stdout:"
        print stdout

    if v > 0 and stderr:
        print '\n> stderr:'
        print stderr

    return stdout, stderr


def setup_mri(mri_sdir, ico=4):
    """
    Prepares an MRI for use in the mne-pipeline:
     - creates symlinks in the bem directory 
     - runs mne_setup_forward_model (see MNE manual section 3.7, p. 25)
    
    The utility needs permission to access the MRI files. In case of a 
    permission error, the following shell command can be used on the mri 
    folder to set permissions appropriately::
    
        $ sudo chmod -R 7700 mri-folder
    
    """
    mri_dir, subject = os.path.split(mri_sdir)
    bemdir = os.path.join(mri_sdir, 'bem')

    # symlinks (MNE-manual 3.6, p. 24 / Gwyneth's Manual X) 
    for name in ['inner_skull', 'outer_skull', 'outer_skin']:
        # can I make path relative by omitting initial bemdir,  ?
        src = os.path.join('watershed', '%s_%s_surface' % (subject, name))
        dest = os.path.join(bemdir, '%s.surf' % name)
        if os.path.exists(dest):
            if os.path.islink(dest):
                if os.path.realpath(dest) == src:
                    pass
                else:
                    logging.debug("replacing symlink: %r" % dest)
                    os.remove(dest)
                    os.symlink(src, dest)
                    # can raise an OSError: [Errno 13] Permission denied
            else:
                raise IOError("%r exists and is no symlink" % dest)
        else:
            os.symlink(src, dest)

    # mne_setup_forward_model
    os.environ['SUBJECTS_DIR'] = mri_dir
    cmd = [get_bin('mne', "mne_setup_forward_model"),
           '--subject', subject,
           '--surf',
           '--ico', ico,
           '--homog']

    _run(cmd)
    # -> creates a number of files in <mri_sdir>/bem



def run_mne_analyze(mri_dir, fif_dir, modal=True):
    """
    invokes mne_analyze (e.g., for manual coregistration)
    
    **Arguments:**
    
    mri_dir : str(path)
        the directory containing the mri data (subjects's mri directory, or 
        fsaverage)
    fif_file : str(path)
        the target fiff file
    modal : bool
        causes the shell to be unresponsive until mne_analyze is closed
    
    
    **Coregistration Procedure:**
    
    (For more information see  MNE-manual 3.10 & 12.11)
    
    #. File > Load Surface: select the subject`s directory and "Inflated"
    #. File > Load digitizer data: select the fiff file
    #. View > Show viewer
       
       a. ``Options``: Switch cortical surface display off, make 
          scalp transparent. ``Done``
       
    #. Adjust > Coordinate Alignment: 
    
       a. set LAP, Nasion and RAP. 
       b. ``Align using fiducials``. 
       c. (``Omit``)
       d. Run ``ICP alignment`` with 20 steps
       e. ``Save default``
    
    this creates a file next to the raw file with the '-trans.fif' extension.
    
    """
    # TODO: use more command line args (manual pdf p. 152)
    os.environ['SUBJECTS_DIR'] = mri_dir
    os.chdir(fif_dir)
    setup_path = get_bin('mne', 'mne_setup_sh')
    bin_path = get_bin('mne', 'mne_analyze')
    p = subprocess.Popen('. %s; %s' % (setup_path, bin_path), shell=True)
    if modal:
        print "Waiting for mne_analyze to be closed..."
        p.wait() # causes the shell to be unresponsive until mne_analyze is closed
#    p = subprocess.Popen(['%s/bin/mne_setup' % _mne_dir],
#                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#    p.wait()
#    a, b = p.communicate('mne_analyze')
#    p = subprocess.Popen('mne_analyze')
# Gwyneth's Manual, XII
## SUBJECTS_DIR is MRI directory according to MNE 17

def run_mne_browse_raw(fif_dir, modal=False):
    os.chdir(fif_dir)
    setup_path = get_bin('mne', 'mne_setup_sh')
    p = subprocess.Popen('. %s; mne_browse_raw' % setup_path, shell=True)
    if modal:
        print "Waiting for mne_browse_raw to be closed..."
        p.wait() # causes the shell to be unresponsive until mne_analyze is closed



def do_forward_solution(paths=dict(rawfif=None,
                                   mri_sdir=None,
                                   fwd='{fif}-fwd.fif',
                                   bem=None,
                                   src=None,
                                   trans=None),
                        overwrite=False, v=1):
    """
    MNE Handbook 3.13
    
    """
    fif_file = paths.get('rawfif')
    mri_sdir = paths.get('mri_sdir')
    fwd_file = paths.get('fwd')
    bem_file = paths.get('bem')
    src_file = paths.get('src')
    trans_file = paths.get('trans')

    mri_dir, mri_subject = os.path.split(mri_sdir)

    fif_name, _ = os.path.splitext(fif_file)
    fwd_file = fwd_file.format(fif=fif_name)

    os.environ['SUBJECTS_DIR'] = mri_dir
    cmd = [get_bin('mne', "mne_do_forward_solution"),
           '--subject', mri_subject,
           '--src', src_file,
           '--bem', bem_file,
#           '--mri', mri_cor_file, # MRI description file containing the MEG/MRI coordinate transformation.
           '--mri', trans_file, # MRI description file containing the MEG/MRI coordinate transformation.
#           '--trans', trans_file, #  head->MRI coordinate transformation (obviates --mri). 
           '--meas', fif_file, # provides sensor locations and coordinate transformation between the MEG device coordinates and MEG head-based coordinates.
           '--fwd', fwd_file, #'--destdir', target_file_dir, # optional 
           '--megonly']

    if overwrite:
        cmd.append('--overwrite')
    elif os.path.exists(fwd_file):
        raise IOError("fwd file at %r already exists" % fwd_file)

    out, err = _run(cmd, v=v)
    if os.path.exists(fwd_file):
        return fwd_file
    else:
        err = "fwd-file not created"
        if v < 1:
            err = os.linesep.join([err, "command out:", out])
        raise RuntimeError(err)


def do_inverse_operator(fwd_file, cov_file, inv_file='{cov}inv.fif',
                        loose=False, fixed=False):
    cov, _ = os.path.splitext(cov_file)
    if cov.endswith('cov'):
        cov = cov[:-3]

    inv_file = inv_file.format(cov=cov)

    cmd = [get_bin('mne', "mne_do_inverse_operator"),
           '--fwd', fwd_file,
           '--meg', # Employ MEG data in the inverse calculation. If neither --meg nor --eeg is set only MEG channels are included.
           '--depth',
           '--megreg', 0.1,
           '--noisecov', cov_file,
           '--inv', inv_file, # Save the inverse operator decomposition here. 
           ]

    if loose:
        cmd.extend(['--loose', loose])
    elif fixed:
        cmd.append('--fixed')

    _, err = _run(cmd)
    if not os.path.exists(inv_file):
        raise RuntimeError(os.linesep.join(["inv-file not created", err]))


# freesurfer---

def _fs_hemis(arg):
    if arg == '*':
        return ['lh', 'rh']
    elif arg in ['lh', 'rh']:
        return [arg]
    else:
        raise ValueError("hemi has to be 'lh', 'rh', or '*' (no %r)" % arg)

def _fs_subjects(arg, mri_dir, exclude=[]):
    if '*' in arg:
        subjects = fnmatch.filter(os.listdir(mri_dir), arg)
        subjects = filter(os.path.isdir, subjects)
        for subject in exclude:
            if subject in subjects:
                subjects.remove(subject)
    else:
        subjects = [arg]
    return subjects


def mri_annotation2label(mri_dir, subject='*', annot='aparc',
                         dest=os.path.join("{sdir}", "label", "{annot}"),
                         hemi='*'):
    """
    Calls ``mri_annotation2label`` (`freesurfer wiki 
    <http://surfer.nmr.mgh.harvard.edu/fswiki/mri_annotation2label>`_)
    
    mri_dir : str(path)
        Path containing mri subject directories (freesurfer's ``SUBJECTS_DIR``).
    
    subject : str
        Name of the subject ('*' uses fnmatch to find folders in ``mri_dir``).
        
    annot : str
        Name of the annotation file (e.g., ``'aparc'``, ...).
    
    dest : str(path)
        Destination for the label files. {sdir}" and "{annot}" are filled in
        appropriately.
        
    hemi : 'lh' | 'rh' | '*'
        Hemisphere to process; '*' proceses both.
    
    """
    hemis = _fs_hemis(hemi)
    subjects = _fs_subjects(subject, mri_dir)

    # progress monitor
    i_max = len(subjects) * len(hemis)
    prog = ui.progress_monitor(i_max, "mri_annotation2label", "")

    for subject in subjects:
        sdir = os.path.join(mri_dir, subject)
        outdir = dest.format(sdir=sdir, annot=annot)

        for hemi in hemis:
            prog.message("Processing: %s - %s" % (subject, hemi))

            cmd = [get_bin("freesurfer", "mri_annotation2label"),
                   '--annotation', annot,
                   '--subject', subject,
                   '--hemi', hemi,
                   '--outdir', outdir,
                   ]

            sout, serr = _run(cmd, cwd=mri_dir)
            prog.advance()



def mri_label2label(mri_dir, src_subject='fsaverage', tgt_subject='*',
                    label='{sdir}/label/*-{hemi}', hemi='*',
                    regmethod='surface'):
    """
    Calls the freesurfer command ``mri_label2label``.
    
    mri_dir : str
        Path containing mri subject directories (freesurfer's ``SUBJECTS_DIR``)
    src_subject : str
        Subject for which the label exists (default: ``'fsaverage'``)
    tgt_subject : str
        subject for which the label should be created (default ``'*'``: all
        subjects except ``src_subjects``)
    label : str
        '{sdir}' and '{hemi}' are filled in using :py:meth:`str.format`; 
        '*' is expanded using fnmatch; 
    hemi : 'lh' | 'rh' | '*'
        only required for ``regmethod=='surface'``
    
    """
    src_sdir = os.path.join(mri_dir, src_subject)

    # find subjects
    subjects = _fs_subjects(tgt_subject, mri_dir, [src_subject])

    # find hemispheres
    hemis = _fs_hemis(hemi)

    # test label pattern (TODO: use glob)
    pattern_head, _ = os.path.split(label)
    if '{hemi}' in pattern_head:
        raise NotImplementedError("{hemi} in directory in %r" % label)
    if '*' in pattern_head:
        raise NotImplementedError("'*' in directory in %r" % label)

    # find labels
    labels = [] # [(label, hemi), ...] paths to labels with {sdir}
    for hemi in hemis:
        label_pattern = label.format(sdir=src_sdir, hemi=hemi)
        label_dir, label_name = os.path.split(label_pattern)
        pattern = os.extsep.join((label_name, 'label'))
        label_names = fnmatch.filter(os.listdir(label_dir), pattern)
        labels.extend((os.path.join(pattern_head, name), hemi) for name in label_names)

    # setup fs
    os.environ['SUBJECTS_DIR'] = mri_dir

    # convert all labels
    prog = ui.progress_monitor(len(subjects) * len(labels), "mri_label2label", "")
    for tgt_subject in subjects:
        tgt_sdir = os.path.join(mri_dir, tgt_subject)
        tgt_dir = pattern_head.format(sdir=tgt_sdir)
        if not os.path.exists(tgt_dir):
            os.mkdir(tgt_dir)

        for label, hemi in labels:
            prog.message("%s" % label.format(sdir=tgt_subject))
            tgt_label = label.format(sdir=tgt_sdir)
            cmd = [get_bin("freesurfer", "mri_label2label"),
                   '--srcsubject', src_subject,
                   '--trgsubject', tgt_subject,
                   '--srclabel', label.format(sdir=src_sdir),
                   '--trglabel', tgt_label,
                   '--regmethod', regmethod,
                   ]
            if regmethod == 'surface':
                cmd.extend(('--hemi', hemi))

            sout, serr = _run(cmd, cwd=mri_dir)
            if not os.path.exists(tgt_label):
                err = "$ mri_label2label failed: %r not created" % tgt_label
                scmd = "Complete commend:\n  %s" % ('\n  '.join(cmd))
                msg = '\n\n'.join((scmd, sout, serr, err))
                raise RuntimeError(msg)

            prog.advance()
