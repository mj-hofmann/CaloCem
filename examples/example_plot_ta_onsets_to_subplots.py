from pathlib import Path
from calocem import Measurement
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


# %% plot the main-peak analysis of each sample into its own subplot

fig, axs = plt.subplots(1, 3, layout="constrained", sharey=True, figsize=(15, 5))

for ax, sample in zip(axs, tam.get_sample_names()):
    tam.get_mainpeak_params(
        regex=sample,
        ax=ax,
        show_plot=True,
    )

    # tune appearance
    ax.set_xlim(0, 60000)
    ax.set_ylim(0, 0.003)
    ax.set_title(sample)

plt.show()
# %%
