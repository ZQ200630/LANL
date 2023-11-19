
import mne

data_path = "/vast/home/qian/freesurfer/"
subjects_dir = data_path + 'data/'
subject = 'subject-1/'
t1_fname = subjects_dir + subject + 'mri/' + 'T1.mgz'
fname = subjects_dir + subject + "surf/" + "rh.white"
rr_mm, tris = mne.read_surface(fname)

src = mne.setup_source_space(
    subject, spacing="oct4", add_dist="patch", subjects_dir=subjects_dir
)

surface = subjects_dir + subject + "bem/" + "brain.surf"
vol_src = mne.setup_volume_source_space(
    subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=False
)  # Just for speed!

conductivity = (0.3, 0.003, 0.3)  # for three layers
model = mne.make_bem_model(
    subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)

from mne.datasets import sample

data_path1 = sample.data_path()

# the raw file containing the channel location + types
sample_dir1 = data_path1 / "MEG" / "sample"
raw_fname1 = sample_dir1 / "sample_audvis_raw.fif"
# The paths to Freesurfer reconstructions
subjects_dir1 = data_path1 / "subjects"
subject1 = "sample"
# The transformation file obtained by coregistration
trans = sample_dir1 / "sample_audvis_raw-trans.fif"


fwd = mne.make_forward_solution(
    raw_fname1,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=True,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)

info = mne.io.read_info(raw_fname1)

selected_label = mne.read_labels_from_annot(
    subject, regexp="caudalmiddlefrontal-lh", subjects_dir=subjects_dir
)[0]
location = "center"  # Use the center of the region as a seed.
extent = 10.0  # Extent in mm of the region.
label = mne.label.select_sources(
    subject, selected_label, location=location, extent=extent, subjects_dir=subjects_dir
)

import numpy as np

tstep = 1.0 / info["sfreq"]
source_time_series = np.sin(2.0 * np.pi * 18.0 * np.arange(100) * tstep) * 10e-9
n_events = 50
events = np.zeros((n_events, 3), int)
events[:, 0] = 100 + 200 * np.arange(n_events)  # Events sample.
events[:, 2] = 1  # All events have the sample id.

source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
source_simulator.add_data(label, source_time_series, events)

print(label)

raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])