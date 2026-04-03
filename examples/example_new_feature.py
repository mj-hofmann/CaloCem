# %%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement

datapath = Path(__file__).parent.parent / "calocem" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# %% use class based approach

# experiments via class
tam = Measurement(
    folder=datapath,
    regex="calorimetry_data_[1-4].csv",
    #regex="excel_example[1-3].xls",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    new_code=True
)

# get sample and information
data = tam.get_data()
info = tam.get_information()


# %%
tam.plot()
plt.show()
# %%
