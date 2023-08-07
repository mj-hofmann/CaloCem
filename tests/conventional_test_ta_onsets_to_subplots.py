import sys
from pathlib import Path

parentfolder = Path(__file__).cwd().parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"myexp[1-4]",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

# init
fig, axs = ta.plt.subplots(1, 3, layout="constrained")

# populate subplots
for i, grad in enumerate([5e-9, 3e-8, 7e-8]):
    
    # get and show onsets
    _, ax = tam.get_peak_onsets(
        ax=axs[i],
        # ax=None,
        # regex="c3a",
        gradient_threshold=grad,
        show_plot=True
        )

    # tune appearance
    # ax.set_xlim(7000, 10000)
    ax.set_xlim(0, 60000)
    ax.set_ylim(0, 0.003)
    ax.set_title(f"{grad:.1e}")