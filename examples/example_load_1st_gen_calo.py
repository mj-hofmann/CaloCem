import matplotlib.pyplot as plt

import sys
from pathlib import Path
import CaloCem.tacalorimetry as ta


datapath = Path(__file__).parent.parent / "CaloCem" / "DATA"
#datapath = parentfolder / "tmp"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"gen1_calofile[2-3].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

ax = tam.plot()
ax.set_xlim(0, 25)
ax.set_ylim(0, 5)
ta.plt.show()
