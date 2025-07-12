# %%
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# from calocem.tacalorimetry import Measurement
from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"
assetpath = Path(__file__).parent.parent / "docs" / "assets"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=".*calorimetry_data_[3-4].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

processparams = ProcessingParameters()
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 5e-12
processparams.gradient_peakdetection.use_largest_width_height = True

# get peak onsets via alternative method

sample_names = tam._data["sample_short"].unique()

for sample_name in sample_names:
    fig, ax = plt.subplots()

    onsets_spline = tam.get_maximum_slope(
        processparams=processparams,
        time_discarded_s=3600,
        exclude_discarded_time=True,
        show_plot=True,
        save_path=assetpath,
        regex=sample_name,
        ax=ax,
        xunit="h",
        xscale="linear",
    )
    # ax.set_xlim(0, 24)
    # ax.set_ylim(0, 0.005)
    ax.set_xscale("log")
    ax.set_title("")
    #ax, ax2 = fig.get_axes()
    handles, labels = ax.get_legend_handles_labels()
    #handles2, labels2 = ax2.get_legend_handles_labels()
    #handles = handles + handles2
    labels = ["Sample", "Gradient"]
    ax.legend(handles, labels, loc="upper right")
    plt.show()
# ta.plt.savefig(assetpath / "example_detect_maximum_slope.png")
# %%
