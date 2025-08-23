#%%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"



# %% plot

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 60
processparams.downsample.apply = True
processparams.downsample.num_points = 2000
# processparams.downsample.section_split = True
# processparams.downsample.section_split_time_s = 3600
processparams.downsample.baseline_weight = 0.1
# processparams.gradient_peakdetection.use_largest_width = True
# processparams.spline_interpolation.apply = True
# processparams.spline_interpolation.smoothing_1st_deriv = 1e-13
# processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10
# processparams.median_filter.apply = True
# processparams.median_filter.size = 5

# experiments via class
tam = Measurement(
    folder=datapath,
    # regex=r".*peak_detection_example[1-7].*|.*calorimetry_data.*",
    regex=r".*peak_detection_example[1-2].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    processparams=processparams
)


#%%
# get peak onsets via alternative method
fig, ax = plt.subplots()
maxslopes = tam.get_maximum_slope(
    processparams=processparams,
    show_plot=True,
    regex=".*example2.*",
    time_discarded_s=3600,
    exclude_discarded_time=True,
    ax=ax,
)
ax.set_xlim(0, 100000)
ax.set_ylim(0, 0.005)
ax.set_xlabel("Time / s")
plt.show()

#%%
fig, ax = plt.subplots()
onsets = tam.get_peak_onset_via_max_slope(
    processparams=processparams,
    show_plot=True,
    #xunit="s",
    regex=".*example1.*",
    time_discarded_s=3600,
    ax=ax,
    intersection="abscissa",
)
ax.set_xlabel("Time / h")
# ax.set_xlim(0, 40)
ax.set_ylim(0, 0.0025)
plt.show()


# %%
mytest = tam.get_ascending_flank_tangent(
    processparams=processparams,
    flank_fraction_start=0.35,
    flank_fraction_end=0.55,
    cutoff_min=75,
    show_plot=True,)
# %%
