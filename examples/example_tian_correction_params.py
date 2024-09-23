from pathlib import Path

import matplotlib.pyplot as plt

import CaloCem.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "CaloCem" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*(insitu_bm).csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# Set Proceesing Parameters
processparams = ta.ProcessingParameters()
processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80
processparams.median_filter.apply = True
processparams.median_filter.size = 15
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-10
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10

# apply tian correction
tam.apply_tian_correction(
    processparams=processparams,
)

df = tam.get_data().copy()

# plot corrected and uncorrected data
fig, ax = plt.subplots()
for name, group in df.groupby("sample_short"):
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g"],
        alpha=0.5,
        linestyle="--",
        label=name[-2:],
    )
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g_tian"],
        color=ax.get_lines()[-1].get_color(),
        label=name[-2:] + " Tian",
    )
ax.set_xlim(0, 15)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Normalized Heat Flow (W/g)")
ax.legend()

plt.savefig(plotpath / "tian_correction.png")


# example with only one tau constant and no smoothing
processparams = ta.ProcessingParameters()
processparams.time_constants.tau1 = 300
processparams.time_constants.tau2 = None
processparams.median_filter.apply = False
processparams.spline_interpolation.apply = False

tam.apply_tian_correction(processparams=processparams)

df_tau = tam.get_data().copy()


fig, ax = plt.subplots()
for (name, group), (name2, group2) in zip(df_tau.groupby("sample_short"), df.groupby("sample_short")):
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g"],
        #alpha=0.5,
        #linestyle="--",
        #label=name[-2:],
        label="no Tian, original"
    )
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g_tian"],
        #color="black",
        label="one tau",
    )
    ax.plot(
        group2["time_s"] / 60,
        group2["normalized_heat_flow_w_g_tian"],
        #color=ax.get_lines()[-1].get_color(),
        # label=name[-2:] + " Tian",
        label="tau1, tau2"
    )
ax.set_xlim(0, 15)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Normalized Heat Flow (W/g)")
ax.legend()
ax.set_title("only one tau constant")
plt.savefig(plotpath / "tian_correction_one_tau.png")


# example without smoothing

processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80
processparams.median_filter.apply = False
processparams.spline_interpolation.apply = False

tam.apply_tian_correction(processparams=processparams)

df2 = tam.get_data()
# plot corrected and uncorrected data
fig, ax = plt.subplots()
for name, group in df2.groupby("sample_short"):
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g"],
        alpha=0.5,
        linestyle="--",
        label=name[-2:],
    )
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g_tian"],
        color=ax.get_lines()[-1].get_color(),
        label=name[-2:] + " Tian",
    )
ax.set_xlim(0, 15)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Normalized Heat Flow (W/g)")
ax.legend()
ax.set_title("No smoothing applied")
plt.savefig(plotpath / "tian_correction_no_smoothing.png")


# example with only median smoothing
processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80
processparams.median_filter.apply = True
processparams.median_filter.size = 15

tam.apply_tian_correction(processparams=processparams)

df3 = tam.get_data()

fig, ax = plt.subplots()
for name, group in df3.groupby("sample_short"):
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g"],
        alpha=0.5,
        linestyle="--",
        label=name[-2:],
    )
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g_tian"],
        color=ax.get_lines()[-1].get_color(),
        label=name[-2:] + " Tian",
    )
ax.set_xlim(0, 15)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Normalized Heat Flow (W/g)")
ax.legend()
ax.set_title("Median Filter applied")
plt.savefig(plotpath / "tian_correction_median_smoothing.png")


# Only Univariate Spline Smoothing

processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80
processparams.median_filter.apply = False
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-10
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10

tam.apply_tian_correction(processparams=processparams)

df4 = tam.get_data()

fig, ax = plt.subplots()
for name, group in df4.groupby("sample_short"):
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g"],
        alpha=0.5,
        linestyle="--",
        label=name[-2:],
    )
    ax.plot(
        group["time_s"] / 60,
        group["normalized_heat_flow_w_g_tian"],
        color=ax.get_lines()[-1].get_color(),
        label=name[-2:] + " Tian",
    )
ax.set_xlim(0, 15)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Normalized Heat Flow (W/g)")
ax.legend()
ax.set_title("Spline Smoothing applied")
plt.savefig(plotpath / "tian_correction_spline_smoothing.png")
