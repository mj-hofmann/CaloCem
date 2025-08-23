"""
Refactored main measurement class for calorimetry data handling.
"""

import logging
import pathlib
import warnings
from typing import Optional, Union

import pandas as pd

from .analysis import (
    ASTMC1679Analyzer,
    AverageSlopeAnalyzer,
    DormantPeriodAnalyzer,
    FlankTangentAnalyzer,
    HeatCalculator,
    OnsetAnalyzer,
    PeakAnalyzer,
    SlopeAnalyzer,
)
from .data_processing import DataCleaner, DataNormalizer, SampleIterator
from .exceptions import AutoCleanException, ColdStartException, DataProcessingException
from .file_io import DataPersistence, FolderDataLoader
from .plotting import SimplePlotter
from .processparams import ProcessingParameters

logger = logging.getLogger(__name__)


class Measurement:
    """
    Refactored class for handling and processing isothermal heat flow calorimetry data.

    This class coordinates file I/O, data processing, analysis, and visualization
    operations while maintaining the same API as the original implementation.
    """

    def __init__(
        self,
        folder: Optional[Union[str, pathlib.Path]] = None,
        show_info: bool = True,
        regex: Optional[str] = None,
        auto_clean: bool = False,
        cold_start: bool = True,
        processparams: Optional[ProcessingParameters] = None,
        new_code: bool = False,
        processed: bool = False,
    ):
        """
        Initialize measurements from folder or existing data.

        Parameters
        ----------
        folder : str or pathlib.Path, optional
            Path to folder containing experimental files
        show_info : bool, optional
            Whether to print informative messages, by default True
        regex : str, optional
            Regex pattern to filter files, by default None
        auto_clean : bool, optional
            Whether to clean data automatically, by default False
        cold_start : bool, optional
            Whether to read from files or use cached data, by default True
        processparams : ProcessingParameters, optional
            Processing parameters, by default None
        new_code : bool, optional
            Flag for new code features, by default False
        processed : bool, optional
            Whether data is already processed, by default False
        """
        # Initialize attributes
        self._data = pd.DataFrame()
        self._info = pd.DataFrame()
        self._data_unprocessed = pd.DataFrame()
        self._metadata = pd.DataFrame()
        self._metadata_id = ""

        # Store configuration
        self._new_code = new_code
        self._processed = processed

        # Setup processing parameters
        if not isinstance(processparams, ProcessingParameters):
            self.processparams = ProcessingParameters()
        else:
            self.processparams = processparams

        # Initialize components
        self._folder_loader = FolderDataLoader()
        self._data_persistence = DataPersistence()
        self._data_cleaner = DataCleaner()
        self._plotter = SimplePlotter()

        # Load data if folder provided
        if folder:
            try:
                if cold_start:
                    self._load_from_folder(folder, regex, show_info)
                else:
                    self._load_from_cache()

                if auto_clean:
                    self._auto_clean_data()

            except Exception as e:
                if show_info:
                    print(f"Error during initialization: {e}")
                if auto_clean:
                    raise AutoCleanException()
                if not cold_start:
                    raise ColdStartException()
                raise

        # Apply downsampling if requested
        if self.processparams.downsample.apply:
            self._apply_adaptive_downsampling()

        # Information message
        if show_info:
            print("================")
            print(
                "Are you missing some samples? Try rerunning with auto_clean=True and cold_start=True."
            )
            print("================")

    def _load_from_folder(
        self, folder: Union[str, pathlib.Path], regex: Optional[str], show_info: bool
    ):
        """Load data from folder using file loader."""
        try:
            self._data, self._info = self._folder_loader.load_from_folder(
                folder, regex, show_info
            )
            self._data_unprocessed = self._data.copy()

            # Save to cache
            self._data_persistence.save_data(self._data, self._info)

        except Exception as e:
            raise DataProcessingException("load_from_folder", e)

    def _load_from_cache(self):
        """Load data from cached pickle files."""
        try:
            if not self._data_persistence.pickle_files_exist():
                raise FileNotFoundError("No pickle files found for cold start")

            self._data, self._info = self._data_persistence.load_data()
            self._data_unprocessed = self._data.copy()

        except Exception as e:
            raise ColdStartException() from e

    def _auto_clean_data(self):
        """Apply automatic data cleaning."""
        try:
            self._data = self._data_cleaner.auto_clean_data(self._data)
        except Exception as e:
            raise AutoCleanException() from e

    def _apply_adaptive_downsampling(self):
        """Apply adaptive downsampling if configured."""
        # TODO: Implement downsampling logic
        logger.info("Downsampling requested but not yet implemented")

    # Data access methods
    def get_data(self) -> pd.DataFrame:
        """Get the processed calorimetry data."""
        return self._data

    def get_information(self) -> pd.DataFrame:
        """Get the measurement information/metadata."""
        return self._info

    def get_metadata(self) -> tuple:
        """Get added metadata and the ID column name."""
        return self._metadata, self._metadata_id

    def get_sample_names(self) -> list:
        """Get list of sample names."""
        return [
            pathlib.Path(str(sample)).stem
            for sample, _ in SampleIterator.iter_samples(self._data)
        ]

    # Plotting methods
    def plot(
        self,
        t_unit: str = "h",
        y: str = "normalized_heat_flow_w_g",
        y_unit_milli: bool = True,
        regex: Optional[str] = None,
        show_info: bool = True,
        ax=None,
    ):
        """Plot the calorimetry data."""
        return self._plotter.plot_data(
            self._data, t_unit, y, y_unit_milli, regex, show_info, ax
        )

    def plot_by_category(
        self,
        categories: str,
        t_unit: str = "h",
        y: str = "normalized_heat_flow_w_g",
        y_unit_milli: bool = True,
    ):
        """Plot data by metadata categories."""
        # Simplified implementation - would need full metadata integration
        logger.warning(
            "plot_by_category requires metadata integration - not fully implemented"
        )
        yield from []

    # Analysis methods
    def get_peaks(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        regex: Optional[str] = None,
        cutoff_min: Optional[float] = None,  # Deprecated parameter
        show_plot: bool = True,
        plt_right_s: float = 2e5,
        plt_top: float = 1e-2,
        ax=None,
        xunit: str = "s",
        plot_labels: Optional[bool] = None,
        xmarker: bool = False,
    ) -> pd.DataFrame:
        """Get DataFrame of peak characteristics."""
        if cutoff_min is not None:
            warnings.warn(
                "The cutoff_min parameter is deprecated. Use ProcessingParameters instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        params = processparams or self.processparams
        analyzer = PeakAnalyzer(params)
        peaks_df = analyzer.get_peaks(self._data, target_col, regex)

        if show_plot and not peaks_df.empty:
            # Simple plotting implementation
            for sample, sample_data in SampleIterator.iter_samples(self._data, regex):
                sample_peaks = peaks_df[
                    peaks_df["sample_short"] == pathlib.Path(str(sample)).stem
                ]
                if not sample_peaks.empty:
                    # Get peak indices relative to sample data
                    import numpy as np

                    peak_indices = np.array(sample_peaks.index.tolist())
                    self._plotter.plot_peaks(
                        sample_data, peak_indices, str(sample), ax, "time_s", target_col
                    )

        return peaks_df

    def get_peak_onsets(
        self,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        time_discarded_s: float = 900,
        rolling: int = 1,
        gradient_threshold: float = 0.0005,
        show_plot: bool = False,
        exclude_discarded_time: bool = False,
        regex: Optional[str] = None,
        ax=None,
    ):
        """Get peak onsets based on gradient threshold."""
        analyzer = OnsetAnalyzer(self.processparams)
        return analyzer.get_peak_onsets(
            self._data,
            target_col,
            age_col,
            time_discarded_s,
            rolling,
            gradient_threshold,
            exclude_discarded_time,
            regex,
        )

    def get_maximum_slope(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        time_discarded_s: float = 900,
        show_plot: bool = False,
        show_info: bool = True,
        exclude_discarded_time: bool = False,
        regex: Optional[str] = None,
        read_start_c3s: bool = False,
        ax=None,
        save_path: Optional[pathlib.Path] = None,
        xscale: str = "log",
        xunit: str = "s",
    ):
        """Find the point in time of the maximum slope."""
        params = processparams or self.processparams
        analyzer = SlopeAnalyzer(params)

        result = analyzer.get_maximum_slope(
            self._data,
            target_col,
            age_col,
            time_discarded_s,
            exclude_discarded_time,
            regex,
            read_start_c3s,
            self._metadata,
        )

        if show_plot and not result.empty:
            for sample, sample_data in SampleIterator.iter_samples(self._data, regex):
                sample_short = pathlib.Path(str(sample)).stem
                sample_result = result[result["sample_short"] == sample_short]
                if not sample_result.empty:
                    self._plotter.plot_slopes(
                        sample_data,
                        sample_result,
                        str(sample_short),
                        ax,
                        age_col,
                        target_col,
                    )

        return result

    def get_peak_onset_via_max_slope(
        self,
        processparams: Optional[ProcessingParameters] = None,
        show_plot: bool = False,
        ax=None,
        regex: Optional[str] = None,
        age_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
        time_discarded_s: float = 900,
        save_path: Optional[pathlib.Path] = None,
        xscale: str = "linear",
        xunit: str = "s",
        intersection: str = "dormant_hf",
    ):
        """Get reaction onset via maximum slope intersection method."""
        params = processparams or self.processparams

        # Get required data
        max_slopes = self.get_maximum_slope(
            params, target_col, age_col, time_discarded_s, False, False, False, regex
        )
        if max_slopes.empty:
            print("No maximum slopes found. Check processing parameters.")
            return pd.DataFrame()

        dormant_hfs = self.get_dormant_period_heatflow(params, regex, show_plot=False)
        if dormant_hfs.empty:
            print("No dormant period heat flows found.")
            return pd.DataFrame()

        # Calculate onsets
        analyzer = OnsetAnalyzer(params)
        result = analyzer.get_peak_onset_via_max_slope(
            self._data, max_slopes, dormant_hfs, intersection, xunit
        )

        # Implement plotting for intersections if show_plot=True
        if show_plot and not result.empty:
            self._plot_onset_intersections(
                result,
                max_slopes,
                dormant_hfs,
                target_col,
                age_col,
                regex,
                intersection,
                xunit,
                time_discarded_s,
                ax,
            )

        return result

    def get_dormant_period_heatflow(
        self,
        processparams: Optional[ProcessingParameters] = None,
        regex: Optional[str] = None,
        cutoff_min: int = 5,
        upper_dormant_thresh_w_g: float = 0.002,
        plot_right_boundary: float = 2e5,
        prominence: float = 1e-3,
        show_plot: bool = False,
    ) -> pd.DataFrame:
        """Get dormant period heat flow characteristics."""
        params = processparams or self.processparams

        # Get peaks first
        peaks = self.get_peaks(params, regex=regex, show_plot=False)

        # Analyze dormant period
        analyzer = DormantPeriodAnalyzer(params)
        return analyzer.get_dormant_period_heatflow(
            self._data, peaks, regex, upper_dormant_thresh_w_g
        )

    def get_astm_c1679_characteristics(
        self,
        processparams: Optional[ProcessingParameters] = None,
        individual: bool = False,
        show_plot: bool = False,
        ax=None,
        regex: Optional[str] = None,
        xscale: str = "log",
        xunit: str = "s",
    ) -> pd.DataFrame:
        """Get characteristics according to ASTM C1679."""
        params = processparams or self.processparams

        # Get peaks first
        peaks = self.get_peaks(params, regex=regex, show_plot=False)

        # Analyze ASTM characteristics
        analyzer = ASTMC1679Analyzer(params)
        return analyzer.get_astm_c1679_characteristics(
            self._data, peaks, individual, regex
        )

    def get_cumulated_heat_at_hours(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_h: float = 4,
        **kwargs,
    ) -> pd.DataFrame:
        """Get cumulated heat flow at specific age."""
        if "cutoff_min" in kwargs:
            cutoff_min = kwargs["cutoff_min"]
            warnings.warn(
                "The cutoff_min parameter is deprecated. Use ProcessingParameters instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            params = processparams or self.processparams
            cutoff_min = params.cutoff.cutoff_min

        return HeatCalculator.get_cumulated_heat_at_hours(
            self._data, target_h, cutoff_min
        )

    def get_average_slope(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        regex: Optional[str] = None,
        show_plot: bool = False,
        ax=None,
        save_path: Optional[pathlib.Path] = None,
        xscale: str = "log",
        xunit: str = "s",
    ) -> pd.DataFrame:
        """Calculate average slope between onset and heat flow maximum."""
        params = processparams or self.processparams

        # Get required data
        max_slopes = self.get_maximum_slope(
            params, target_col, age_col, regex=regex, show_plot=False
        )
        onsets = self.get_peak_onset_via_max_slope(params, regex=regex, show_plot=False)

        if max_slopes.empty or onsets.empty:
            logger.warning("Cannot calculate average slopes - missing required data")
            return pd.DataFrame()

        # Calculate average slopes
        analyzer = AverageSlopeAnalyzer(params)
        return analyzer.get_average_slope(
            self._data, max_slopes, onsets, target_col, age_col, regex
        )

    def get_ascending_flank_tangent(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        flank_fraction_start: float = 0.2,
        flank_fraction_end: float = 0.8,
        window_size: float = 0.1,
        cutoff_min: Optional[float] = None,
        show_plot: bool = False,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
    ) -> pd.DataFrame:
        """
        Determine tangent to ascending flank of peak by averaging over sections.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters, by default None
        target_col : str
            Column containing heat flow data
        age_col : str
            Column containing time data
        flank_fraction_start : float
            Start of flank section as fraction of peak height (0-1)
        flank_fraction_end : float
            End of flank section as fraction of peak height (0-1)
        window_size : float
            Size of averaging window as fraction of flank time range
        cutoff_min : float, optional
            Initial cutoff time in minutes to ignore from analysis. If None,
            uses processparams.cutoff.cutoff_min. The default is None.
        show_plot : bool
            Whether to plot the results
        regex : str
            Regex to filter samples
        plotpath : pathlib.Path, optional
            Path to save plots

        Returns
        -------
        pd.DataFrame
            DataFrame with tangent characteristics for each sample
        """
        params = processparams or self.processparams
        analyzer = FlankTangentAnalyzer(params)

        result = analyzer.get_ascending_flank_tangent(
            self._data,
            target_col,
            age_col,
            flank_fraction_start,
            flank_fraction_end,
            window_size,
            cutoff_min,
            regex,
        )

        if show_plot and not result.empty:
            # Determine the actual cutoff time used
            cutoff_time_min = (
                cutoff_min if cutoff_min is not None else params.cutoff.cutoff_min
            )
            self._plot_flank_tangent_results(
                result, target_col, age_col, regex, plotpath, cutoff_time_min
            )

        return result

    def _plot_flank_tangent_results(
        self,
        results: pd.DataFrame,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        cutoff_time_min: Optional[float] = None,
    ):
        """Plot flank tangent analysis results using SimplePlotter."""
        try:
            for _, result_row in results.iterrows():
                sample = result_row["sample"]
                sample_short = result_row["sample_short"]

                # Get sample data
                sample_data = self._data[self._data["sample"] == sample]
                if sample_data.empty:
                    continue

                # Apply cutoff if specified
                if cutoff_time_min is not None:
                    cutoff_seconds = cutoff_time_min * 60
                    sample_data = sample_data[sample_data[age_col] >= cutoff_seconds]

                # Create a DataFrame with just this result for plotting
                single_result = pd.DataFrame([result_row])

                # Use SimplePlotter to create the plot
                self._plotter.plot_flank_tangent(
                    sample_data,
                    single_result,
                    sample_short,
                    ax=None,
                    age_col=age_col,
                    target_col=target_col,
                    cutoff_time_min=cutoff_time_min,
                )

                if plotpath:
                    plot_file = plotpath / f"flank_tangent_{sample_short}.png"
                    import matplotlib.pyplot as plt

                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")

                # Show the plot
                import matplotlib.pyplot as plt

                plt.show()

        except Exception as e:
            logger.error(f"Error plotting flank tangent results: {e}")
            # Fallback to simple message
            print(f"Plotting failed: {e}")

    def _plot_onset_intersections(
        self,
        onsets: pd.DataFrame,
        max_slopes: pd.DataFrame,
        dormant_hfs: pd.DataFrame,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        intersection: str = "dormant_hf",
        xunit: str = "s",
        time_discarded_s: float = 900,
        ax=None,
    ):
        """Plot onset intersection analysis results using SimplePlotter."""
        try:
            for _, onset_row in onsets.iterrows():
                sample = onset_row["sample"]

                # Get sample data
                sample_data = self._data[self._data["sample_short"] == sample]
                if sample_data.empty:
                    continue

                # Apply time filtering if specified
                if time_discarded_s > 0:
                    sample_data = sample_data[
                        sample_data[age_col] >= time_discarded_s
                    ]  # Keep all data for visualization

                # Use SimplePlotter to create the plot
                self._plotter.plot_onset_intersections(
                    sample_data,
                    max_slopes,
                    dormant_hfs,
                    onsets,
                    sample,
                    ax=ax,
                    age_col=age_col,
                    target_col=target_col,
                    intersection=intersection,
                    xunit=xunit,
                    cutoff_time_min=None,  # Could be added as parameter if needed
                )

                # Show the plot
                import matplotlib.pyplot as plt

                if not ax:
                    plt.show()

        except Exception as e:
            logger.error(f"Error plotting onset intersections: {e}")
            print(f"Plotting failed: {e}")

    # Data manipulation methods
    def normalize_sample_to_mass(
        self, sample_short: str, mass_g: float, show_info: bool = True
    ):
        """Normalize heat flow values to a specific mass."""
        self._data = DataNormalizer.normalize_sample_to_mass(
            self._data, sample_short, mass_g, show_info
        )

    def add_metadata_source(self, file: str, sample_id_column: str):
        """Add metadata from external source."""
        # TODO: Implement metadata loading
        logger.warning("add_metadata_source not yet implemented in refactored version")

    def remove_pickle_files(self):
        """Remove pickle cache files."""
        self._data_persistence.remove_pickle_files()

    # Private utility methods
    def _iter_samples(self, regex: Optional[str] = None):
        """Iterate over samples - compatibility method."""
        return SampleIterator.iter_samples(self._data, regex)
