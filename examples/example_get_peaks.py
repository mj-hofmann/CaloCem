# %%
import sys
from pathlib import Path

import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*peak_detection.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)
#%%

processparams = ta.ProcessingParameters()

processparams.peak_prominence = 1e-5
processparams.spline_interpolation = {
    "apply": True,
    "smoothing_1st_deriv": 1e-11,
    "smoothing_2nd_deriv": 1e-10,
}

peaks_found = tam.get_peaks(processparams, plt_right_s=3e5)


# %%