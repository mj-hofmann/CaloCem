import sys
from pathlib import Path
import matplotlib.pyplot as plt
# parentfolder = Path(__file__).cwd().parent
# sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"LED.*",
    # regex="calo_sharp_sd3.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

tam.plot()
plt.ylim(0, 5)
plt.show()
plt.close()

# %% plot

# get peak onsets via alternative method
onsets_spline = tam.get_maximum_slope(
    show_plot=True,
    use_first=False,
    use_largest_width=True,
    time_discarded_s=3600,
    spline_smoothing=1e-13,
    prominence=5e-9,
    window=21,
    # window=155,
    width=50,
    distance=100,
    rel_height=0.25,
)
# onsets_roll = tam.get_maximum_slope(show_plot=True, use_first=True, time_discarded_s=3600, rolling="1min")

# plot
ta.plt.plot(range(len(onsets_spline)), onsets_spline["time_s"], "ro:", label="spline")
# ta.plt.plot(range(len(onsets_roll)), onsets_roll["time_s"], "bp:", label="rolling")

# %%
