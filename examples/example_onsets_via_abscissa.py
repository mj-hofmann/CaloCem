#%%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"



# %% plot

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 90
processparams.downsample.apply = True
processparams.downsample.num_points = 3000
# processparams.downsample.section_split = True
# processparams.downsample.section_split_time_s = 3600
processparams.downsample.baseline_weight = 0.1

# experiments via class
tam = Measurement(
    folder=datapath,
    # regex=r".*peak_detection_example[1-7].*|.*calorimetry_data.*",
    regex=r".*peak_detection_example[1-4].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    processparams=processparams
)



#%%
fig, ax = plt.subplots()
onsets = tam.get_peak_onset_via_slope(
    processparams=processparams,
    show_plot=True,
    plot_type="mean",
    regex=".*example[2].*",
    #ax=ax,
)
ax.set_xlabel("Time / h")
# ax.set_xlim(0, 40)
# ax.set_ylim(0, 0.0025)
plt.show()

