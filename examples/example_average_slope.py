#%%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*peak_detection_example[1,3].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

processparams = ProcessingParameters()


processparams.gradient_peakdetection.use_largest_width = True
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-13
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10
processparams.median_filter.apply = True
processparams.median_filter.size = 5
#%%
# get peak onsets via alternative method
fig, ax = plt.subplots()
maxslopes = tam.get_maximum_slope(
    processparams=processparams,
    show_plot=True,
    regex=".*example1.*",
    ax=ax,
)
ax.set_xlabel("Time / s")
plt.show()

#%%
fig, ax = plt.subplots()
slopes = tam.get_average_slope(
    processparams=processparams,
    show_plot=True,
    xunit="s",
    regex=".*example3.*",
    ax=ax,
    #intersection="abscissa",
)
ax.set_xlabel("Time / s")
ax.set_xlim(0, 1e5)
ax.set_ylim(0, 0.006)
plt.show()


# %%
