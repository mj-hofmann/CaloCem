
#%%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

tianparams = ta.TianParameters()
tianparams.tau_values = {"tau1": 230, "tau2":70}
tianparams.median_filter["apply"] = True
tianparams.median_filter["size"] = 5
tianparams.spline_interpolation["apply"] = True
tianparams.spline_interpolation["smoothing_1st_deriv"] = 1e-11
tianparams.spline_interpolation["smoothing_2nd_deriv"] = 1e-10

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*(insitu_bm).*.csv",
    show_info=True,
    auto_clean=False,
    cold_start=False,
)

tam.apply_tian_correction(
    tianparams=tianparams,
)

#%%

fig, ax = plt.subplots()
ax.plot(tam._data["time_s"], tam._data["normalized_heat_flow_w_g"], alpha=0.5, linestyle=":")
ax.plot(tam._data["time_s"], tam._data["normalized_heat_flow_w_g_tian"])
ax.set_xlim(0, 1000)
# %%
