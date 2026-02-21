# %%
from pathlib import Path

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# Load example data
tam = Measurement(
    folder=datapath,
    regex=r".*flank_detection.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

# Configure processing
processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 60
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 5e-12
processparams.slope_analysis.flank_fraction_start = 0.2
processparams.slope_analysis.flank_fraction_end = 0.9
processparams.slope_analysis.first_ascending_fraction_of_max = 0.6
processparams.slope_analysis.first_ascending_range_method = "delta"
processparams.slope_analysis.first_ascending_delta_y_w_g = 1e-3
processparams.slope_analysis.flexible = 0.0

# Optional: sample-specific overrides via regex rules (priority: lower value = earlier)
processparams.add_sample_param_rule(
    r".*flank_detection[1,2].*",
    {
        "slope_analysis": {
            "first_ascending_delta_y_w_g": 1e-3,
            "flexible": 0.5,
        }
    },
    priority=10,
)
# processparams.add_sample_param_rule(
#     r".*flank_detection4.*",
#     {
#         "slope_analysis": {
#             "first_ascending_delta_y_w_g": 4.0e-4,
#             "flexible": 1.4,
#         }
#     },
#     priority=10,
# )

# Alternative: use fraction-based range detection
# processparams.slope_analysis.first_ascending_range_method = "fraction"

# Run unified main-peak analysis in ascending mode
mainpeak = tam.get_mainpeak_params(
    processparams=processparams,
    show_plot=True,
    plot_type="mean",  # ignored in method="ascending"
    method="ascending",
)

# Select and display columns from the new first-ascending-slope analysis
first_ascending_cols = [
    "sample_short",
    "range_method_for_first_ascending_slope",
    "delta_y_w_g_for_first_ascending_slope",
    "flexible_for_first_ascending_slope",
    "delta_y_multiplier_for_first_ascending_slope",
    "delta_y_effective_w_g_for_first_ascending_slope",
    "fraction_of_max_for_first_ascending_slope",
    "normalized_heat_flow_w_g_threshold_for_first_ascending_slope",
    "gradient_of_first_ascending_slope_to_fraction_of_max",
    "onset_time_s_from_first_ascending_slope",
    "onset_time_s_from_first_ascending_slope_abscissa",
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