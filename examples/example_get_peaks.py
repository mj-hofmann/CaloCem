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
    regex=r".*peak_detection_example[3-4].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)
# %%

processparams = ProcessingParameters()
processparams.peakdetection.prominence = 1e-4

fig, axs = plt.subplots(1, 2)

peaks_found = tam.get_peaks(processparams, plt_right_s=3e5, ax=axs[0], regex="4", show_plot=True)
axs[0].set_xlim(0, 100000)
peaks_found2 = tam.get_peaks(processparams, plt_right_s=3e5, ax=axs[1], regex="3", show_plot=True)
axs[1].set_xlim(0, 100000)

plt.savefig(plotpath / "example_get_peaks.png", dpi=300)

df = peaks_found[0]
df = df.iloc[:,[0,4,5,10]]

df.to_csv(plotpath / "example_get_peaks.csv", index=False)

# %%
fig, ax = plt.subplots()
tam.get_peaks(
    processparams=processparams,
    ax=ax,
    show_plot=True,
    regex="4",
    xunit="h",
    plot_labels=True,
)
plt.show()
# %%
