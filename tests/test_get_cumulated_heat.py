import pathlib

import pysnooper
import pytest

from TAInstCalorimetry import tacalorimetry


#
# run "time check" on each sample file
# see https://www.youtube.com/watch?v=DhUpxWjOhME (~ at 12:20)
# to use the the test:
#       1) install the module locally with "pip install -e ."
#       2) run "pytest" from shell


@pytest.mark.parametrize(
    "target_h",
    [(1), ([1, 2, 3, 4])],  # only one time  # list of times
)
@pysnooper.snoop()
def test_get_cumulated_heat(target_h):

    # path
    path = pathlib.Path().cwd() / "TAInstCalorimetry" / "DATA"

    # init object
    tam = tacalorimetry.Measurement(path, auto_clean=False, show_info=True)

    # get cumulated heats
    cumulated_heats = tam.get_cumulated_heat_at_hours(target_h=target_h, cutoff_min=10)

    # check
    assert isinstance(cumulated_heats, tacalorimetry.pd.DataFrame)
