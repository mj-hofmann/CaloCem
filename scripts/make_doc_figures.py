"""
Create the figures of docs/quantification.md.

The code mirrors the snippets on that page. Run it after changing the plots:

    MPLBACKEND=Agg uv run python scripts/make_doc_figures.py
"""

import pathlib

import matplotlib.pyplot as plt

from calocem import Measurement, ProcessingParameters

ROOT = pathlib.Path(__file__).parent.parent
ASSETS = ROOT / "docs" / "assets"
FIGSIZE = (7, 4.5)
DPI = 150

tam = Measurement(
    folder=ROOT / "calocem" / "DATA",
    regex=r"calorimetry_data_[12]\.csv",
    auto_clean=False,
    show_info=False,
)

# Peaks
processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 60
processparams.peakdetection.prominence = 1e-4

fig, ax = plt.subplots(figsize=FIGSIZE)
tam.get_peaks(processparams, show_plot=True, ax=ax)
ax.set_xlim(0, 100000)
ax.set_ylim(0, 0.004)
fig.tight_layout()
fig.savefig(ASSETS / "example_get_peaks.png", dpi=DPI)
plt.close(fig)

# Main hydration peak
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-13
processparams.plotting.time_unit = "hours"
processparams.plotting.heat_unit = "mW"

fig, ax = plt.subplots(figsize=FIGSIZE)
tam.get_mainpeak_params(
    processparams, show_plot=True, plot_type="max", regex="data_1", ax=ax
)
fig.tight_layout()
fig.savefig(ASSETS / "example_detect_maximum_slope.png", dpi=DPI)
plt.close(fig)

# ASTM C1679
fig, ax = plt.subplots(figsize=FIGSIZE)
tam.get_astm_c1679_characteristics(processparams, show_plot=True, xunit="h", ax=ax)
ax.set_xlim(0, 48)
ax.set_ylim(0, 0.004)
fig.tight_layout()
fig.savefig(ASSETS / "example_astm_c1679.png", dpi=DPI)
plt.close(fig)
