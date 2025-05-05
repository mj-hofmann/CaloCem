import matplotlib.pyplot as plt

import sys
from pathlib import Path
from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters


datapath = Path(__file__).parent.parent / "calocem" / "DATA"

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 60
processparams.cutoff.cutoff_max = 96 * 60 



# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r"excel_example[2-4].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    processparams=processparams
)

ax = tam.plot()
ax.set_xlim(0, 25)
ax.set_ylim(0, 5)
plt.show()

# print last heat flow values per sample
for name, sample in tam._data.groupby("sample_short"):
    print(f"Last time: {sample.time_s.iloc[-1]}")