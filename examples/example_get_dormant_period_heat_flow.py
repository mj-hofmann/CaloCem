# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import CaloCem.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "CaloCem" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=".*peak_detect.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot


processparams = ta.ProcessingParameters()

processparams.peak_prominence = 1e-5
# processparams.spline_interpolation = {
#     "apply": True,
#     "smoothing_1st_deriv": 1e-11,
#     "smoothing_2nd_deriv": 1e-10,
# }
# processparams.median_filter = {
#     "apply": True,
#     "size": 5,
# }
processparams.cutoff_min = 60

# get peak onsets via alternative method
onsets_spline = tam.get_dormant_period_heatflow(
    processparams=processparams,
    show_plot=True,
    plot_right_boundary=3e5
)

# %%
