#%%
from pathlib import Path
import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"
assetpath = Path(__file__).parent.parent / "docs" / "assets"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=".*calorimetry_data_5.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot


processparams = ProcessingParameters()
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-12

# get peak onsets via alternative method
# fig, ax = ta.plt.subplots()
onsets_spline = tam.get_maximum_slope(
    processparams=processparams,
    show_plot=True,
    save_path=assetpath,
    #ax = ax
)
# ta.plt.savefig(assetpath / "example_detect_maximum_slope.png")
# %%