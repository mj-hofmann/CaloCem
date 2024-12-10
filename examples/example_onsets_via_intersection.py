from pathlib import Path

import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*peak_detec.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

processparams = ProcessingParameters()

processparams.gradient_peakdetection.prominence = 1e-10
processparams.gradient_peakdetection.width = 200
processparams.gradient_peakdetection.use_largest_width = True
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1.5e-12
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10
processparams.median_filter.apply = True
processparams.median_filter.size = 25

# get peak onsets via alternative method
maxslopes = tam.get_maximum_slope(
    processparams=processparams,
    show_plot=True,
)

onsets = tam.get_peak_onset_via_max_slope(
    processparams=processparams,
    show_plot=True
)


# %%
