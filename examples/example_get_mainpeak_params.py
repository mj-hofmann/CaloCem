#%%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem import Measurement
from calocem import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"
metadatapath = Path(__file__).parent.parent / "calocem" / "METADATA" / "metadata_dummy.csv"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*peak_detection_example[1].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    metadata_path=metadatapath,
    metadata_id_column="file_name"
)


    # %% plot

processparams = ProcessingParameters()


processparams.gradient_peakdetection.use_largest_width = True
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-13
processparams.spline_interpolation.smoothing_2nd_deriv = 1e-10
processparams.median_filter.apply = True
processparams.median_filter.size = 5
processparams.cutoff.cutoff_min = 75
processparams.slope_analysis.flank_fraction_start = 0.4
processparams.slope_analysis.flank_fraction_end = 0.6
processparams.plotting.figsize = (5, 4)
processparams.plotting.time_unit = "hours"
processparams.plotting.heat_unit = "mW"
processparams.plotting.show_plot_title = True
processparams.plotting.plot_title = ['cem:', 'cement_name', 'cement_amount_g']
processparams.plotting.legend_pos = "outside"
processparams.plotting.xlims = (0, 75)
processparams.plotting.ylims = (0, 3)  # Set x-axis limits
# %%
mainpeak = tam.get_mainpeak_params(
    processparams=processparams,
    show_plot=True,
    save_plot=True,
    plot_type="mean"
    #regex=".*example3.*",
)

# %%
