# %%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem import Measurement, ProcessingParameters


datapath = Path(__file__).parent.parent / "calocem" / "DATA"


# %%
tam = Measurement(
    folder=datapath,
    regex=".*bm.*",
    cold_start=True,
    auto_clean=False,
)


def _plot_tian(tam, processparams, title, xlim_h=1.0):
    """Overlay raw and Tian-corrected heat flow for all samples."""
    fig, ax = plt.subplots()
    for sample, data in tam.get_data().groupby("sample_short"):
        (line,) = ax.plot(
            data["time_s"] / 3600,
            data["normalized_heat_flow_w_g"] * 1e3,
            alpha=0.5,
            linestyle=":",
            label=f"{sample} (raw)",
        )
        ax.plot(
            data["time_s"] / 3600,
            data["normalized_heat_flow_w_g_tian"] * 1e3,
            color=line.get_color(),
            label=f"{sample} (Tian)",
        )
    ax.set_xlabel("Time / h")
    ax.set_ylabel("Normalized heat flow / mW g$^{-1}$")
    ax.set_xlim(0, xlim_h)
    ax.set_ylim(-50,)
    ax.legend(fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# %%
# --- single time constant, no smoothing ---
# tau2 = None selects first-order correction: hf_corr = dHF/dt * tau1 + HF
processparams = ProcessingParameters()
processparams.time_constants.tau1 = 300
processparams.time_constants.tau2 = None

tam.apply_tian_correction(processparams)
_plot_tian(tam, processparams, "Single time constant — no smoothing")

# %%
# --- dual time constants, no smoothing ---
# second-order: hf_corr = dHF/dt*(tau1+tau2) + d²HF/dt²*tau1*tau2 + HF
processparams = ProcessingParameters()
processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80

tam.apply_tian_correction(processparams)
_plot_tian(tam, processparams, "Dual time constants — no smoothing")

# %%
# --- dual time constants + median filter ---
processparams = ProcessingParameters()
processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80
processparams.median_filter.apply = True
processparams.median_filter.size = 15

tam.apply_tian_correction(processparams)
_plot_tian(tam, processparams, "Dual time constants — median filter (size=15)")

# %%
# --- dual time constants + spline smoothing ---
processparams = ProcessingParameters()
processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-10
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10

tam.apply_tian_correction(processparams)
_plot_tian(tam, processparams, "Dual time constants — spline smoothing")

# %%
# --- dual time constants + median filter + spline smoothing ---
processparams = ProcessingParameters()
processparams.time_constants.tau1 = 240
processparams.time_constants.tau2 = 80
processparams.median_filter.apply = True
processparams.median_filter.size = 15
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-10
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10

tam.apply_tian_correction(processparams)
_plot_tian(tam, processparams, "Dual time constants — median filter + spline smoothing")
