#%%
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=".*peak_detection_example.*",
    show_info=True,
    auto_clean=False,
    cold_start=False,
)


# %% plot


processparams = ta.ProcessingParameters()

processparams.gradient_peak_prominence = 1e-8
processparams.use_largest_gradient_peak_width_height = False
processparams.use_first_detected_gradient_peak = True
processparams.gradient_peak_height = 1e-9
processparams.spline_interpolation = {
    "apply": True,
    "smoothing_1st_deriv": 1e-11,
    "smoothing_2nd_deriv": 1e-10,

}
processparams.median_filter = {
    "apply": True,
    "size": 15,
}


# get peak onsets via alternative method
onsets_spline = tam.get_maximum_slope(
    processparams=processparams,
    show_plot=True,
)

# %%