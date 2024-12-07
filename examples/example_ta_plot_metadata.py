#%%
from pathlib import Path
import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement

parentfolder = Path(__file__).parent.parent
datapath = parentfolder / "calocem" / "DATA"
plotpath = parentfolder / "docs" / "assets"
testpath = parentfolder / "tests"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*data_[1-5].csv",
    show_info=True,
    auto_clean=False,
    cold_start=False,
)


#%%
fig, ax = plt.subplots()
tam.plot(ax=ax)
ax.set_xlim(0,24)
ax.set_ylim(0,5)
plt.show()

# add metadata and average
tam.add_metadata_source(
    testpath / "mini_metadata.csv", sample_id_column="experiment_nr"
)
tam.average_by_metadata(group_by="cement_name")

fig, ax = plt.subplots()
tam.plot(ax=ax)
ax.set_xlim(0, 24)
ax.set_ylim(0, 5)
plt.show()
# %%
