# %%
from pathlib import Path
import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*peak_detection_example[3].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)
# %%

processparams = ta.ProcessingParameters()
processparams.peakdetection.prominence = 1e-4

fig, ax = ta.plt.subplots()

peaks_found = tam.get_peaks(processparams, plt_right_s=3e5, ax=ax, show_plot=True)
ax.set_xlim(0, 100000)
ta.plt.savefig(plotpath / "example_get_peaks.png", dpi=300)


peaks_found[0].to_csv(plotpath / "example_get_peaks.csv", index=False)

# %%
