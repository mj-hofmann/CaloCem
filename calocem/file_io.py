"""
File I/O operations for calorimetry data.
"""

import logging
import pathlib
import pickle
import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import pandas as pd
from scipy import integrate

from calocem import utils

from .exceptions import FileReadingException

logger = logging.getLogger(__name__)


class FileReader(ABC):
    """Abstract base class for file readers."""

    @abstractmethod
    def can_read(self, file_path: pathlib.Path) -> bool:
        """Check if this reader can handle the file."""
        pass

    @abstractmethod
    def read_data(
        self, file_path: pathlib.Path, show_info: bool = True
    ) -> pd.DataFrame:
        """Read experimental data from file."""
        pass

    @abstractmethod
    def read_info(
        self, file_path: pathlib.Path, show_info: bool = True
    ) -> pd.DataFrame:
        """Read metadata/info from file."""
        pass


class CSVReader(FileReader):
    """Handles CSV file reading."""

    def __init__(self, processed: bool = False):
        """
        Initialize CSV reader.

        Parameters
        ----------
        processed : bool
            Whether the data is already processed
        """
        self.processed = processed

    def can_read(self, file_path: pathlib.Path) -> bool:
        return file_path.suffix.lower() == ".csv"

    def read_data(
        self, file_path: pathlib.Path, show_info: bool = True
    ) -> pd.DataFrame:
        """Read calorimetry data from CSV file."""
        try:
            if self.processed:
                # For processed data, read directly with headers
                data = self._read_processed_csv(file_path, show_info)
            else:
                # Try different reading strategies for raw data
                data = self._read_comma_separated(file_path, show_info)
                if data is None:
                    data = self._read_tab_separated(file_path, show_info)

            if data is None:
                raise FileReadingException(file_path, "No valid CSV format found")

            logger.info(f"✓ reading {file_path} successful.")
            return data

        except Exception as e:
            logger.error(f"✗ reading {file_path} FAILED: {e}")
            raise FileReadingException(file_path, e)

    def _read_processed_csv(
        self, file_path: pathlib.Path, show_info: bool = True
    ) -> pd.DataFrame:
        """
        Read already processed CSV data with standard headers.

        This method assumes the CSV file has proper headers and is already
        in the expected format (like the output from previous processing).
        """
        try:
            # Read CSV with headers - assume comma separated
            data = pd.read_csv(file_path, sep=",", header=0)

            # Ensure sample information is present
            if "sample" not in data.columns:
                data["sample"] = str(file_path)
            if "sample_short" not in data.columns:
                data["sample_short"] = file_path.stem

            if show_info:
                print(
                    f"Read processed data from {file_path.name} with {len(data)} rows"
                )

            return data

        except Exception as e:
            if show_info:
                print(f"Error reading processed CSV {file_path.name}: {e}")
            raise

    def read_info(
        self, file_path: pathlib.Path, show_info: bool = True
    ) -> pd.DataFrame:
        """Read metadata from CSV file."""
        try:
            data = pd.read_csv(
                file_path, header=None, sep="No meaningful separator", engine="python"
            )

            # Look for metadata in header rows
            if data[0].str.contains("Sample mass").any():
                info_row = data[data[0].str.contains("Sample mass")].index[0]
                info = data.iloc[info_row : info_row + 1, :2].copy()
                info.columns = ["parameter", "value"]
            else:
                # Create empty info DataFrame
                info = pd.DataFrame(columns=["parameter", "value"])

            # Add sample information
            info["sample"] = str(file_path)
            info["sample_short"] = file_path.stem

            return info

        except Exception as e:
            logger.warning(f"Could not read info from {file_path}: {e}")
            # Return minimal info
            info = pd.DataFrame(columns=["parameter", "value"])
            info["sample"] = str(file_path)
            info["sample_short"] = file_path.stem
            return info

    def _scan_csv_structure(
        self, file_path: pathlib.Path
    ) -> Optional[tuple]:
        """
        Scan the file line-by-line to determine its structure.

        Returns (header_row, footer_row, reaction_start_time) where:
          - header_row: line index of the column-name row
          - footer_row: line index of the "Data series" footer (or None)
          - reaction_start_time: float offset to subtract from time_s (or None)

        Returns None if the file appears to be tab-separated.
        """
        header_row = None
        footer_row = None
        reaction_start_time = None

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i == 0 and "\t" in line:
                    return None  # tab-separated — hand off to _read_tab_separated

                line_lower = line.lower()

                # Extract non-zero reaction-start offset before data starts
                if reaction_start_time is None and "reaction start" in line_lower:
                    try:
                        t = float(line.split(",")[0].strip().strip('"'))
                        if t != 0.0:
                            reaction_start_time = t
                    except (ValueError, IndexError):
                        pass

                # Header row: first line containing both "time" and "heat"
                if header_row is None and "time" in line_lower and "heat" in line_lower:
                    header_row = i

                # Footer row: "Data series" statistics block — stop reading here
                if header_row is not None and "data series" in line_lower:
                    footer_row = i
                    break

        if header_row is None:
            return None

        return header_row, footer_row, reaction_start_time

    def _read_comma_separated_fast(
        self, file_path: pathlib.Path, show_info: bool
    ) -> Optional[pd.DataFrame]:
        """
        Fast CSV reading via direct pd.read_csv after a cheap line-scan.

        Skips pre-header metadata rows, excludes the trailing "Data series"
        statistics block, and reuses the existing tidy_colnames /
        remove_unnecessary_data / convert_df_to_float pipeline.
        """
        structure = self._scan_csv_structure(file_path)
        if structure is None:
            return None  # tab-separated

        header_row, footer_row, reaction_start_time = structure

        data = pd.read_csv(
            file_path,
            sep=",",
            header=None,          # keep first post-skip row as data so tidy_colnames works
            skiprows=range(header_row),
            engine="c",
            on_bad_lines="skip",  # gracefully handle footer rows with wrong column count
        )

        # Drop the "Data series" statistics footer if present
        if footer_row is not None:
            data = data[~data[0].astype(str).str.contains("Data series", case=False, na=False)]
            data = data.reset_index(drop=True)

        data = utils.tidy_colnames(data)
        if data is None:
            return None

        data = utils.remove_unnecessary_data(data)
        data = utils.convert_df_to_float(data).copy()

        # Apply reaction-start offset found during scan
        if reaction_start_time is not None:
            data = data.assign(time_s=data["time_s"] - reaction_start_time)

        # Handle reaction_start_time_s column (some file variants)
        try:
            if (
                "reaction_start_time_s" in data.columns
                and not data["reaction_start_time_s"].isna().all()
            ):
                rs = data["reaction_start_time_s"].dropna().iloc[0]
                data = data.assign(time_s=data["time_s"] - rs)
        except Exception:
            pass

        data = data.query("time_s > 0").reset_index(drop=True)
        data = utils.add_sample_info(data, file_path)

        return data

    def _read_comma_separated(
        self, file_path: pathlib.Path, show_info: bool
    ) -> Optional[pd.DataFrame]:
        """Read comma-separated CSV data — fast path with legacy fallback."""
        data = self._read_comma_separated_fast(file_path, show_info)
        if data is not None:
            return data

        # Legacy fallback: row-by-row Python parsing via parse_rowwise_data
        return self._read_comma_separated_legacy(file_path, show_info)

    def _read_comma_separated_legacy(
        self, file_path: pathlib.Path, show_info: bool
    ) -> Optional[pd.DataFrame]:
        """Legacy row-by-row CSV reader kept as fallback."""
        try:
            data = pd.read_csv(
                file_path, header=None, sep="No meaningful separator", engine="python"
            )

            # Check for tab-separation
            if "\t" in str(data.at[0, 0]):
                return None  # Try tab-separated instead

            # Look for potential index indicating in-situ-file
            reaction_start_row = None
            if data[0].str.contains("Reaction start").any():
                reaction_start_row = data[0].str.contains("Reaction start").idxmax()

            data = utils.parse_rowwise_data(data)
            data = utils.tidy_colnames(data)
            data = utils.remove_unnecessary_data(data)
            data = utils.convert_df_to_float(data).copy()

            if reaction_start_row:
                try:
                    reaction_start = float(data.at[reaction_start_row, "time_s"])
                    data = data.assign(time_s=data["time_s"] - reaction_start)
                except Exception:
                    pass

            # Check for "in-situ" sample and reset if needed
            try:
                if (
                    "reaction_start_time_s" in data.columns
                    and not data["reaction_start_time_s"].isna().all()
                ):
                    reaction_start = data["reaction_start_time_s"].dropna().iloc[0]
                    data = data.assign(time_s=data["time_s"] - reaction_start)
            except Exception:
                pass

            # Restrict to positive time values
            data = data.query("time_s > 0").reset_index(drop=True)

            # Add sample information
            data = utils.add_sample_info(data, file_path)

            return data

        except Exception:
            return None

    def _read_tab_separated(
        self, file_path: pathlib.Path, show_info: bool
    ) -> Optional[pd.DataFrame]:
        """Read tab-separated CSV data."""
        try:
            raw = pd.read_csv(file_path, sep="\t", header=None)
            data = raw.copy()

            # Get sample mass if available
            mass = None
            try:
                # the convention is to have the sample mass in the 4th column, first and the second index is found
                # and then the value is extracted
                mass_index = data.index[data.iloc[:, 3].notna()]
                if not mass_index.empty:
                    mass_str = str(data.iloc[mass_index[1], 3])
                mass = float(re.findall(r"[\d.]+", mass_str)[0])
            except (IndexError, ValueError):
                if show_info:
                    print(f"No sample mass found in {file_path}")

            # Get reaction start time if available
            t0 = None
            try:
                # reaction start string is in the 3rd
                _helper = data[data.iloc[:, 2].str.lower() == "reaction start"].head(1)
                t0 = float(_helper[0].values[0])

            except Exception:
                pass

            # Remove all-NaN columns and restrict to first two columns
            data = data.dropna(how="all", axis=1).iloc[:, :2]

            # Set column names
            try:
                data.columns = ["time_s", "heat_flow_mw"]
            except ValueError:
                return None

            # Get data rows (skip header)
            data = data.loc[3:, :].reset_index(drop=True)

            # Convert data types
            #data["time_s"] = data["time_s"].astype(float)
            data = data.assign(time_s=pd.to_numeric(data["time_s"], errors="coerce"))
            
            # data["heat_flow_mw"] = data["heat_flow_mw"].apply(
            #     lambda x: float(str(x).replace(",", "."))
            # )
            data = data.assign(
                heat_flow_mw=data["heat_flow_mw"].astype(str).str.replace(",", ".").astype(float)
            )

            # Convert units
            #data["heat_flow_w"] = data["heat_flow_mw"] / 1000
            data = data.assign(heat_flow_w=data["heat_flow_mw"] / 1000)

            # Calculate cumulative heat
            data = data.assign(heat_j=integrate.cumulative_trapezoid(
                data["heat_flow_w"], x=data["time_s"], initial=0
            ))

            # Remove intermediate column
            del data["heat_flow_mw"]

            # Apply time offset
            if t0:
                data = data.assign(time_s=data["time_s"] - t0)
                # data["time_s"] = data["time_s"] - t0

            # Calculate normalized values if mass is available
            if mass:
                data = data.assign(
                    normalized_heat_flow_w_g=data["heat_flow_w"] / mass
                )
                data = data.assign(
                    normalized_heat_j_g=data["heat_j"] / mass
                )

            # Restrict to non-negative time
            data = data.query("time_s >= 0").reset_index(drop=True)

            # Add sample information
            data = data.assign(
                sample=str(file_path),
                sample_short=file_path.stem
            )

            data = utils.convert_df_to_float(data)

            return data

        except Exception:
            return None


class XLSReader(FileReader):
    """Handles XLS file reading."""

    def can_read(self, file_path: pathlib.Path) -> bool:
        return file_path.suffix.lower() == ".xls"

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names from XLS data."""
        # Replace init timestamp
        df.iloc[0, 0] = "time"

        df = utils.tidy_colnames(df)

        return df

    def read_data(
        self, file_path: pathlib.Path, show_info: bool = True
    ) -> pd.DataFrame:
        """Read calorimetry data from XLS file."""
        try:
            xl = pd.ExcelFile(file_path)

            # Try to read from "RawData" sheet
            try:
                df_data = pd.read_excel(xl, "Raw data", header=None)

                df_data = self.clean_column_names(df_data)
                # Remove columns with too many NaNs
                df_data = df_data.dropna(axis=1, thresh=3)

                # cut away the initial three rows
                df_data = df_data.iloc[2:, :].reset_index(drop=True)

                # Convert columns to float
                for col in df_data.columns:
                    try:
                        # df_data[col] = pd.to_numeric(df_data[col], errors="coerce")
                        df_data = df_data.assign(**{col: pd.to_numeric(df_data[col], errors="coerce")})
                    except Exception:
                        pass

                # Add sample information
                df_data["sample"] = str(file_path)
                df_data["sample_short"] = file_path.stem

                logger.info(f"✓ reading {file_path} successful.")
                return df_data

            except Exception as e:
                if show_info:
                    print(f"Could not read data from {file_path}: {e}")
                logger.error(f"✗ reading {file_path} FAILED: {e}")
                raise FileReadingException(file_path, e)

        except Exception as e:
            raise FileReadingException(file_path, e)

    def read_info(
        self, file_path: pathlib.Path, show_info: bool = True
    ) -> pd.DataFrame:
        """Read metadata from XLS file."""
        try:
            xl = pd.ExcelFile(file_path)

            try:
                # Try to read from "SampleData" sheet
                info = pd.read_excel(xl, "SampleData").T
                info.columns = info.iloc[0]
                info = info.drop(info.index[0])

                # Add sample information
                info["sample"] = str(file_path)
                info["sample_short"] = file_path.stem

                return info

            except Exception:
                # Try alternative sheet names or create empty
                try:
                    sheet_names = xl.sheet_names
                    info_sheet = [
                        s
                        for s in sheet_names
                        if isinstance(s, str)
                        and ("info" in s.lower() or "sample" in s.lower())
                    ]
                    if info_sheet:
                        info = pd.read_excel(xl, info_sheet[0])
                    else:
                        info = pd.DataFrame()
                except Exception:
                    info = pd.DataFrame()

                # Add sample information
                info["sample"] = str(file_path)
                info["sample_short"] = file_path.stem

                return info

        except Exception as e:
            logger.warning(f"Could not read info from {file_path}: {e}")
            # Return minimal info
            info = pd.DataFrame()
            info["sample"] = str(file_path)
            info["sample_short"] = file_path.stem
            return info


class FileReaderFactory:
    """Factory for creating appropriate file readers."""

    def __init__(self, processed: bool = False):
        """
        Initialize file reader factory.

        Parameters
        ----------
        processed : bool
            Whether the data is already processed
        """
        self.processed = processed
        self.readers = [CSVReader(processed=processed), XLSReader()]

    def get_reader(self, file_path: Union[str, pathlib.Path]) -> Optional[FileReader]:
        """Get appropriate reader for file."""
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        for reader in self.readers:
            if reader.can_read(file_path):
                return reader
        return None

    def read_file(
        self, file_path: Union[str, pathlib.Path], show_info: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read both data and info from file."""
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        reader = self.get_reader(file_path)
        if reader is None:
            raise FileReadingException(
                file_path, "No reader available for this file type"
            )

        data = reader.read_data(file_path, show_info)
        info = reader.read_info(file_path, show_info)

        return data, info


class DataPersistence:
    """Handles data persistence (pickle files)."""

    def __init__(self, base_path: Optional[pathlib.Path] = None):
        if base_path is None:
            base_path = pathlib.Path.cwd()
        self.data_pickle_path = base_path / "_data.pickle"
        self.info_pickle_path = base_path / "_info.pickle"

    def save_data(self, data: pd.DataFrame, info: pd.DataFrame):
        """Save data and info to pickle files."""
        try:
            with open(self.data_pickle_path, "wb") as f:
                pickle.dump(data, f)
            with open(self.info_pickle_path, "wb") as f:
                pickle.dump(info, f)
            logger.info("Data saved to pickle files successfully")
        except Exception as e:
            logger.error(f"Failed to save data to pickle files: {e}")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and info from pickle files."""
        try:
            with open(self.data_pickle_path, "rb") as f:
                data = pickle.load(f)
            with open(self.info_pickle_path, "rb") as f:
                info = pickle.load(f)
            logger.info("Data loaded from pickle files successfully")
            return data, info
        except Exception as e:
            logger.error(f"Failed to load data from pickle files: {e}")
            raise

    def pickle_files_exist(self) -> bool:
        """Check if pickle files exist."""
        return self.data_pickle_path.exists() and self.info_pickle_path.exists()

    def remove_pickle_files(self):
        """Remove pickle files."""
        try:
            if self.data_pickle_path.exists():
                self.data_pickle_path.unlink()
            if self.info_pickle_path.exists():
                self.info_pickle_path.unlink()
            logger.info("Pickle files removed successfully")
        except Exception as e:
            logger.error(f"Failed to remove pickle files: {e}")
            raise


class FolderDataLoader:
    """Loads data from a folder containing calorimetry files."""

    def __init__(self, processed: bool = False):
        """
        Initialize folder data loader.

        Parameters
        ----------
        processed : bool
            Whether the data is already processed
        """
        self.file_factory = FileReaderFactory(processed=processed)

    def load_from_folder(
        self,
        folder: Union[str, pathlib.Path],
        regex: Optional[str] = None,
        show_info: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load all supported files from a folder."""
        if isinstance(folder, str):
            folder = pathlib.Path(folder)

        all_data = []
        all_info = []

        # Get all supported files
        supported_extensions = {".xls", ".csv"}
        files = [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        # Apply regex filter if provided
        if regex:
            files = [f for f in files if re.match(regex, f.name)]

        for file_path in files:
            if show_info:
                print(f"Reading {file_path.name}.")

            try:
                data, info = self.file_factory.read_file(file_path, show_info)
                all_data.append(data)
                all_info.append(info)
            except FileReadingException as e:
                logger.warning(f"Skipping file {file_path}: {e}")
                if show_info:
                    print(f"Warning: Could not read {file_path.name}")
                continue

        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_info = pd.concat(all_info, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
            combined_info = pd.DataFrame()

        return combined_data, combined_info
