# %%
from pathlib import Path

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# Load example data
tam = Measurement(
    folder=datapath,
    regex=r".*flank_detection3.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

# Configure processing
processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 60
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 5e-12
processparams.slope_analysis.flank_fraction_start = 0.35
processparams.slope_analysis.flank_fraction_end = 0.55
processparams.slope_analysis.first_ascending_fraction_of_max = 0.5

# Run unified main-peak analysis (includes first ascending slope to 20% of max)
mainpeak = tam.get_mainpeak_params(
    processparams=processparams,
    show_plot=True,
    plot_type="mean",
)

# Select and display columns from the new first-ascending-slope analysis
first_ascending_cols = [
    "sample_short",
    "fraction_of_max_for_first_ascending_slope",
    "normalized_heat_flow_w_g_threshold_for_first_ascending_slope",
    "gradient_of_first_ascending_slope_to_fraction_of_max",
    "first_ascending_slope_start_time_s",
    "first_ascending_slope_end_time_s",
    "normalized_heat_flow_w_g_at_first_ascending_slope_start",
    "normalized_heat_flow_w_g_at_first_ascending_slope_end",
    "number_of_points_for_first_ascending_slope",
]

available_cols = [col for col in first_ascending_cols if col in mainpeak.columns]
print(mainpeak[available_cols])

# Optional: plot mean-slope analysis with tangent overlay
# tam.get_mainpeak_params(
#     processparams=processparams,
#     show_plot=True,
#     plot_type="mean",
# )

# %%