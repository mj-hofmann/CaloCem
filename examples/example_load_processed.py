#%%
from pathlib import Path
from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters


datapath = Path(__file__).parent.parent / "calocem" / "DATA"
#datapath = parentfolder / "tmp"

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 60
processparams.slope_analysis.window_size = 0.3
processparams.slope_analysis.flank_fraction_start = 0.25
processparams.slope_analysis.flank_fraction_end = 0.75


# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r"calorimetry_data.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    new_code=True,
    processed=False,
    processparams=processparams
)

print(tam._data.head(10))

#%%

onsets = tam.get_peak_onset_via_slope(
    processparams=processparams,
    show_plot=True,
    plot_type="mean",
    #regex=".*example[1-7].*",
    #ax=ax,
)
# %%
