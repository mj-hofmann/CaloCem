
#%%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "CaloCem" / "DATA"
# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*(insitu_bm).*.csv",
    show_info=True,
    auto_clean=False,
    cold_start=False,
)

#%%
processparams = ProcessingParameters()
processparams.time_constants.tau1 = 230
processparams.time_constants.tau2 = 80
processparams.median_filter.apply = True
processparams.median_filter.size = 15
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-10
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10

tam.apply_tian_correction(
    processparams=processparams,
)



fig, ax = plt.subplots()
for name, group in tam._data.groupby("sample_short"):
    ax.plot(group["time_s"], group["normalized_heat_flow_w_g"], alpha=0.5, linestyle="--")
    ax.plot(group["time_s"], group["normalized_heat_flow_w_g_tian"], color=ax.get_lines()[-1].get_color())
ax.set_xlim(0, 500)
# %%
