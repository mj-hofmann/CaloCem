import sys
from pathlib import Path

import matplotlib.pyplot as plt

parentfolder = Path(__file__).parent.parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"calo_c3s.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

# get peak onsets via alternative method
fig, ax = plt.subplots()
tam.set_savgol_parameters(window=11, polynom=3)
tam.set_spline_parameters(smoothing=1e-9)
tam.set_peak_detection_parameters(rel_height=.5, height=1e-8, width=200)

onsets, ax = tam.get_peak_onset_via_max_slope(
    show_plot=True, cutoff_min=60, prominence=1e-9, ax=ax
)

print(onsets)
# %%
tam.plot()
# %%
