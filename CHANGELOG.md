# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
