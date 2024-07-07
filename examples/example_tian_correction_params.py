# %%
from pathlib import Path

import matplotlib.pyplot as plt

import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*(insitu_bm).*.csv",
    show_info=True,
    auto_clean=False,
    cold_start=False,
)

# %%
processparams = ta.ProcessingParameters()
processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 240

processparams.median_filter["apply"] = True
processparams.median_filter["size"] = 15
processparams.spline_interpolation["apply"] = True
processparams.spline_interpolation["smoothing_1st_deriv"] = 1e-10
processparams.spline_interpolation["smoothing_2nd_deriv"] = 1e-10

tam.apply_tian_correction(
    processparams=processparams,
)

df = tam.get_data()

fig, ax = plt.subplots()
for name, group in df.groupby("sample_short"):
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g"],
        alpha=0.5,
        linestyle="--",
        label="raw",
    )
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g_tian"],
        color=ax.get_lines()[-1].get_color(),
        label="tian-corrected",
    )
ax.set_xlim(0, 15)
ax.set_xlabel("Time [min]")
ax.set_ylabel("Heat flow [W/g]")
ax.legend()
# %%
