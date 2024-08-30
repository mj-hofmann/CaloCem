# %%
from pathlib import Path
import CaloCem.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "CaloCem" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*peak_detection_example[3-4].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)
# %%

processparams = ta.ProcessingParameters()
processparams.peakdetection.prominence = 1e-4

fig, axs = ta.plt.subplots(1, 2)

peaks_found = tam.get_peaks(processparams, plt_right_s=3e5, ax=axs[0], regex="4", show_plot=True)
axs[0].set_xlim(0, 100000)
peaks_found2 = tam.get_peaks(processparams, plt_right_s=3e5, ax=axs[1], regex="3", show_plot=True)
axs[1].set_xlim(0, 100000)

ta.plt.savefig(plotpath / "example_get_peaks.png", dpi=300)

df = peaks_found[0]
df = df.iloc[:,[0,4,5,10]]

df.to_csv(plotpath / "example_get_peaks.csv", index=False)

# %%
