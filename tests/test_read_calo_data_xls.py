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
#
@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("OPC_1.xls", 499440),
        ("OPC_2.xls", 336000),
    ],
)
@pysnooper.snoop()
def test_last_time_entry(test_input, expected):

    # path
    path = pathlib.Path().cwd() / "TAInstCalorimetry" / "DATA"

    # get data
    data = tacalorimetry.Measurement()._read_calo_data_xls(path / test_input)

    # get "last time for a file
    last_time = int(data.tail(1)["time_s"])

    # actual test
    assert last_time == int(expected)