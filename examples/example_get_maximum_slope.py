#%%
from pathlib import Path
import matplotlib.pyplot as plt

import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=".*calorimetry_data_5.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot


processparams = ta.ProcessingParameters()
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-12

# get peak onsets via alternative method
fig, ax = ta.plt.subplots()
onsets_spline = tam.get_maximum_slope(
    processparams=processparams,
    show_plot=True,
    ax = ax
)

# %%