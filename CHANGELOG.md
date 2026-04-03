# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-03

### Breaking Changes

- **`calocem.tacalorimetry` has been removed.** The legacy monolithic
  `Measurement` implementation (`tacalorimetry.py`, ~3 400 lines) has been
  retired. The module now exists as a stub that raises a `FutureWarning` on
  import and forwards to the refactored implementation. The old and new
  implementations are **not guaranteed to produce identical numerical results**.
  Users who require the previous behaviour should pin to `calocem<0.3.0`.

  Migrate your imports:

  ```python
  # old
  from calocem.tacalorimetry import Measurement

  # new — preferred
  from calocem import Measurement

  # new — explicit
  from calocem.measurement import Measurement
  ```

### Added

- **Public package API.** `Measurement` and `ProcessingParameters` are now
  importable directly from the top-level package:
  ```python
  from calocem import Measurement, ProcessingParameters
  ```

### Fixed

- Eliminated all pandas Copy-on-Write `FutureWarning` and `DeprecationWarning`
  occurrences triggered by chained assignment patterns in `utils.py` and
  `tacalorimetry.py`. The package is now silent on pandas 2.x and compatible
  with the Copy-on-Write behaviour that becomes the default in pandas 3.0.
- `float()` called on a single-element `Series` replaced with
  `float(series.iloc[0])` to silence pandas `DeprecationWarning`.
- Removed stray `print("hallo")` debug statement from the data loading path.

## [0.2.5] - 2026-04-03

### Fixed
- `ipykernel` moved to dev dependencies — it is no longer installed as part of
  a normal `pip install calocem`. Users who need it for notebooks should install
  it explicitly (`pip install ipykernel`).
- Replaced `logging.basicConfig()` call at import time with a module-level
  logger. CaloCem no longer writes `CaloCem.log` to the working directory on
  import or hijacks the host application's log configuration.

### Internal
- `/.cache` added to `.gitignore` to stop the mkdocs-git-committers cache from
  being tracked.
- Manual exploration scripts (`script_test_*.py`) moved from `tests/` to
  `examples/` where the other example scripts live. They are not pytest tests
  and were silently collected without asserting anything.
- Removed unused live `import pysnooper` from `tests/test_read_calo_data_xls.py`.

## [0.2.4] - 2025-09-28

- See repository history for changes prior to this changelog.
