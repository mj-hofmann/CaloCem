#%%
from pathlib import Path
from calocem import Measurement, ProcessingParameters


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
    regex=r"process.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    new_code=True,
    processed=True,
    processparams=processparams
)

print(tam.get_data().head(10))

#%%
tam.plot()
slopes = tam.get_maximum_slope(processparams=processparams, show_plot=True)
# %%
