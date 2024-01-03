import sys
from pathlib import Path

parentfolder = Path(__file__).cwd().parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"myexp[1-4]",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

# get peak onsets via alternative method
onsets_spline = tam.get_maximum_slope(show_plot=False)
onsets_roll = tam.get_maximum_slope(show_plot=False, time_discarded_s=3600, rolling="15min")

# plot
ta.plt.plot(range(len(onsets_spline)), onsets_spline["time_s"], "ro:", label="spline")
ta.plt.plot(range(len(onsets_roll)), onsets_roll["time_s"], "bp:", label="rolling")

ta.plt.legend()