import pathlib
import shutil

import pytest

from calocem.measurement import Measurement


DATA_DIR = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
SAMPLE = "excel_example4.xls"


def _run_measurement(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    return Measurement(DATA_DIR, regex=SAMPLE, show_info=False, **kwargs)


def test_default_does_not_write_pickles(tmp_path, monkeypatch):
    _run_measurement(tmp_path, monkeypatch)
    assert not (tmp_path / "_data.pickle").exists()
    assert not (tmp_path / "_info.pickle").exists()


def test_save_cache_true_writes_pickles(tmp_path, monkeypatch):
    _run_measurement(tmp_path, monkeypatch, save_cache=True)
    assert (tmp_path / "_data.pickle").exists()
    assert (tmp_path / "_info.pickle").exists()


def test_cold_start_false_reads_existing_cache(tmp_path, monkeypatch):
    primed = _run_measurement(tmp_path, monkeypatch, save_cache=True)
    primed_data = primed.get_data()

    reloaded = Measurement(DATA_DIR, regex=SAMPLE, show_info=False, cold_start=False)
    assert reloaded.get_data().equals(primed_data)


def test_save_cache_false_with_cold_start_false_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from calocem.exceptions import ColdStartException

    with pytest.raises(ColdStartException):
        Measurement(DATA_DIR, regex=SAMPLE, show_info=False, cold_start=False)
