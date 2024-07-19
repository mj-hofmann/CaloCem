#%%
from pathlib import Path
import matplotlib.pyplot as plt

import TAInstCalorimetry.tacalorimetry as ta

parentfolder = Path(__file__).parent.parent
datapath = parentfolder / "TAInstCalorimetry" / "DATA"
plotpath = parentfolder / "docs" / "assets"
testpath = parentfolder / "tests"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*data_[1-5].csv",
    show_info=True,
    auto_clean=False,
    cold_start=False,
)


#%%
tam.add_metadata_source(
    testpath / "mini_metadata.csv", sample_id_column="experiment_nr"
)

fig, ax = plt.subplots()
tam.plot(ax=ax)
ax.set_xlim(0,24)
ax.set_ylim(0,5)
plt.show()

tam.average_by_metadata(group_by="cement_name")

fig, ax = plt.subplots()
tam.plot(ax=ax)
ax.set_xlim(0, 24)
ax.set_ylim(0, 5)
plt.show()
# %%
