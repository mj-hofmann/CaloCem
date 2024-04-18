# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.signal import find_peaks, peak_prominences, peak_widths

import TAInstCalorimetry.tacalorimetry as ta
import TAInstCalorimetry.utils as utils

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    # regex=r"calo_sharp_sd.csv",
    regex=r"myexp9.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

# %%
df = tam._data

df_query = df.query("time_s > 3600").copy()

df_query["dydx"], _ = utils.calculate_smoothed_heatflow_derivatives(
    df_query, window=35, spline_smoothing_1st=2e-13
)
# the_peaks, _ = find_peaks(df_query["dydx"], width=20, distance=50, prominence=5e-9, rel_height=1)
the_peaks, _ = find_peaks(
    df_query["dydx"],
    width=15,
    distance=100,
    prominence=1e-9,
    height=1e-9,
    rel_height=0.05,
)
peak_width_list = peak_widths(df_query["time_s"], the_peaks, rel_height=1)
# peak_prominence_list = peak_prominences(df_query["dydx"], the_peaks)

# %%
fig, ax = plt.subplots()
ax.plot(df_query["time_s"], df_query["normalized_heat_flow_w_g"] * 1)
ax.plot(df_query["time_s"], df_query["dydx"] * 1e4)
# plot found peaks
ax.plot(
    df_query["time_s"].values[the_peaks], df_query["dydx"].values[the_peaks] * 1e4, "x"
)
plt.hlines(*peak_width_list[1:], color="black")
ax.set_ylim(-0.001, 0.003)
# ax.set_xlim(200000, 210000)
# %%
