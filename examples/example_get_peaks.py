# %%
from pathlib import Path
from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters
from matplotlib import pyplot as plt

datapath = Path(__file__).parent.parent / "calocem" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*peak_detection_example[3].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)
# %%

processparams = ProcessingParameters()
processparams.peakdetection.prominence = 1e-4
processparams.cutoff.cutoff_min = 75


fig, ax = plt.subplots()
tam.get_peaks(
    processparams=processparams,
    ax=ax,
    show_plot=True,
    # regex="4",
    xunit="h",
    plot_labels=True,
    xmarker=True,
)
plt.show()
# %%
