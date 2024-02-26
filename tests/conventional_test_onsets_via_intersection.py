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
    regex=r"myexp9.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

# get peak onsets via alternative method
fig, ax = plt.subplots()
onsets, ax = tam.get_peak_onset_via_max_slope(
    show_plot=True, cutoff_min=15, prominence=1e-4, ax=ax
)

print(onsets)
# %%
tam.plot()
# %%
