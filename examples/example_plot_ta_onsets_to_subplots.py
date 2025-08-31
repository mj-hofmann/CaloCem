from pathlib import Path
from calocem.tacalorimetry import Measurement
import matplotlib.pyplot as plt

parentfolder = Path(__file__).parent.parent

datapath = parentfolder / "calocem" / "DATA"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*data_[1-3].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

# init
fig, axs = plt.subplots(1, 3, layout="constrained", sharey=True)

# populate subplots
for i, grad in enumerate([5e-9, 3e-8, 7e-8]):
    
    # get and show onsets
    _, ax = tam.get_peak_onsets(
        ax=axs[i],
        gradient_threshold=grad,
        show_plot=True
        )

    # tune appearance
    # ax.set_xlim(7000, 10000)
    ax.set_xlim(0, 60000)
    ax.set_ylim(0, 0.003)
    ax.set_title(f"{grad:.1e}")
# %%
