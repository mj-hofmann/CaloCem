import matplotlib.pyplot as plt

import sys
from pathlib import Path
import TAInstCalorimetry.tacalorimetry as ta


datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"
#datapath = parentfolder / "tmp"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*1st.*.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

fig, ax = plt.subplots()
tam.plot()
ax.set_xlim(0, 25)
ax.set_ylim(0, 5)
ta.plt.show()
