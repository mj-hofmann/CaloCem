"""
Refactored main measurement class for calorimetry data handling.
"""

import logging
import pathlib
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    Class for handling and processing isothermal heat flow calorimetry data.

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
            Processing parameters, by default None. If None, the default parameters will be used
        new_code : bool, optional
            Flag for new code features, by default False
        processed : bool, optional
            Whether data is already processed, i.e., if a .csv file is used which was processed  by Calocem. By default False
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
        self._folder_loader = FolderDataLoader(processed=processed)
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
        xscale: str = "linear",
        xunit: str = "s",
    ):
        """Find the point in time of the maximum slope."""
        params = processparams or self.processparams

        time_discarded_s = (
            params.cutoff.cutoff_min * 60 if params.cutoff.cutoff_min else 0
        )
        analyzer = SlopeAnalyzer(params)

        result = analyzer.get_maximum_slope(
            self._data,
            target_col,
            age_col,
            time_discarded_s,
            exclude_discarded_time,
            regex,
            # read_start_c3s,
            # self._metadata,
        )

        if show_plot and not result.empty:
            for sample, sample_data in SampleIterator.iter_samples(self._data, regex):
                sample_short = pathlib.Path(str(sample)).stem
                sample_result = result[result["sample_short"] == sample_short]
                sample_result = sample_result[
                    sample_result[age_col] >= time_discarded_s
                ]
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

    def get_mainpeak_params(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        show_plot: bool = False,
        plot_type: str = "mean",
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        ax=None,
    ) -> pd.DataFrame:
        """
        Unified method that calculates BOTH maximum and mean slope onset analyses.

        This method performs both slope-based analysis approaches simultaneously:
        - Maximum slope: Uses single point with maximum gradient for onset determination
        - Mean slope: Uses averaged slope over flank windows for onset determination

        Both results are returned in a single DataFrame with all slope values and onsets.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters, by default None
        target_col : str
            Column containing heat flow data. The default is 'normalized_heat_flow_w_g'.
        age_col : str
            Column containing time data. The default is 'time_s'.
        show_plot : bool
            Whether to plot the results
        plot_type : str
            Type of plot to show: 'max', 'mean',
            - 'max': Shows only maximum slope analysis plot
            - 'mean': Shows only mean slope (flank tangent) analysis plot
        regex : str, optional
            Regex to filter samples
        plotpath : pathlib.Path, optional
            Path to save plots
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on

        Returns
        -------
        pd.DataFrame
            Comprehensive DataFrame with both max and mean slope results including:
            - Gradients and curvatures at max slope
            - Gradients of mean slope
            - Onset times from both methods
            - Normalized heat flow and heat values at key points
            - Dormant period heat flow values
            - ASTM C1679 characteristic values
        
        Examples
        --------
        >>> measurement = Measurement(folder="data/")
        >>> mainpeak_params = measurement.get_mainpeak_params(
        ...     processparams=ProcessingParameters(),
        ...     show_plot=False,
        ...     plot_type="mean",
        ... )
        """
        params = processparams or self.processparams

        max_slope_results = self._calculate_max_slope_analysis(
            params,
            target_col,
            age_col,
            regex,
        )

        mean_slope_results = self._calculate_mean_slope_analysis(
            params,
            target_col,
            age_col,
            regex,
        )

        dormant_minimum_heatflow = self.get_dormant_period_heatflow(
            params, regex, show_plot=False
        )

        astm_values = self.get_astm_c1679_characteristics(params, individual=True, show_plot=False, regex=regex)

        # Merge results into comprehensive DataFrame
        combined_results = self._merge_slope_results(
            max_slope_results, mean_slope_results, dormant_minimum_heatflow, astm_values
        )

        # Plot if requested
        if show_plot and not (mean_slope_results.empty or max_slope_results.empty):
            self._plot_combined_slope_analysis(
                combined_results,
                params,
                target_col,
                age_col,
                plot_type,
                regex,
                plotpath,
                ax,
            )
            if not ax:
                plt.show()
        elif mean_slope_results.empty:
            # logger.warning("No slope analysis results to plot.")
            print("No mean slope analysis obtained - check the processing parameters.")

        elif max_slope_results.empty:
            print(
                "No maximum slope analysis obtained - check the processing parameters."
            )

        return combined_results

    def _calculate_max_slope_analysis(
        self,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        regex: Optional[str],
    ) -> pd.DataFrame:
        """Calculate maximum slope analysis and return structured results."""
        # Get required data
        max_slope_analyzer = SlopeAnalyzer(params)
        max_slopes = max_slope_analyzer.get_maximum_slope(
            self._data,
            target_col,
            age_col,
            regex,
        )

        if max_slopes.empty:
            logger.warning("No maximum slopes found. Check processing parameters.")
            return pd.DataFrame()

        dormant_hfs = self.get_dormant_period_heatflow(params, regex, show_plot=False)
        if dormant_hfs.empty:
            logger.warning("No dormant period heat flows found.")
            return pd.DataFrame()

        # Calculate onsets
        analyzer = OnsetAnalyzer(params)
        onsets = analyzer.get_peak_onset_via_max_slope(
            self._data,
            max_slopes,
            dormant_hfs,  # intersection, xunit
        )

        # Structure results with consistent naming
        results = []
        for _, slope_row in max_slopes.iterrows():
            sample = slope_row.get("sample", slope_row.get("sample_short", ""))
            sample_short = slope_row.get("sample_short", slope_row.get("sample", ""))

            onset_row = (
                onsets[onsets["sample_short"] == sample_short]
                if not onsets.empty
                else pd.DataFrame()
            )
            onset_time = (
                onset_row.iloc[0]["onset_time_s"] if not onset_row.empty else None
            )

            # get normalized_heat_j_g at onset_time
            if onset_time and not pd.isna(onset_time):
                onset_j_g = np.interp(
                    onset_time,
                    self._data[age_col],
                    self._data["normalized_heat_j_g"],
                )
            else:
                onset_j_g = None

            result_data = {
                "sample": sample,
                "sample_short": sample_short,
                "gradient_from_max_slope": slope_row.get("gradient", 0),
                "curvature_at_max_slope": slope_row.get("curvature", 0),
                "max_slope_time_s": slope_row.get("time_s", 0),
                "normalized_heat_flow_w_g_at_max_slope": slope_row.get(
                    "normalized_heat_flow_w_g", 0
                ),
                "normalized_heat_j_g_at_max_slope": slope_row.get("normalized_heat_j_g", 0),
                "normalized_heat_j_g_at_onset_time_max_slope": onset_j_g,
                "onset_time_s_from_max_slope": onset_time,
                "onset_time_min_max_slope": onset_time / 60 if onset_time else None,
                "onset_time_s_max_slope_abscissa": (
                    onset_row.iloc[0]["onset_time_s_abscissa"]
                    if not onset_row.empty
                    else None
                ),
            }
            results.append(result_data)

        return pd.DataFrame(results)

    def _calculate_mean_slope_analysis(
        self,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        regex: Optional[str],
    ) -> pd.DataFrame:
        """Calculate mean slope (flank tangent) analysis and return structured results."""
        analyzer = FlankTangentAnalyzer(params)

        # Get flank tangent results
        tangent_results = analyzer.get_ascending_flank_tangent(
            self._data,
            target_col,
            age_col,
            regex,
        )

        if tangent_results.empty:
            logger.warning("No flank tangent results found.")
            return pd.DataFrame()

        results = []
        for _, row in tangent_results.iterrows():
            sample = row.get("sample", row.get("sample_short", ""))
            sample_short = row.get("sample_short", row.get("sample", ""))

            # onset by intersection with tangent to dormant period
            onset_time = row.get("x_intersection_dormant", row.get("tangent_time_s", 0))

            result_data = {
                "sample": sample,
                "sample_short": sample_short,
                "gradient_of_mean_slope": row.get("tangent_slope", 0),
                "mean_slope_time_s": row.get("tangent_time_s", 0),
                "normalized_heat_flow_w_g_at_mean_slope": row.get("tangent_value", 0),
                "normalized_heat_j_g_at_mean_slope": row.get("tangent_j_g", 0),
                "onset_time_s_from_mean_slope": onset_time,
                "onset_time_min_from_mean_slope": onset_time / 60 if onset_time else None,
                "onset_time_s_from_mean_slope_abscissa": row.get("x_intersection", 0),
                "normalized_heat_at_onset_time_mean_slope_abscissa_j_g": row.get("x_intersection_j_g", 0),
                "normalized_heat_at_onset_time_mean_slope_dormant_j_g": row.get("x_intersection_dormant_j_g", 0),
                "flank_start_value": row.get("flank_start_value", 0),
                "flank_end_value": row.get("flank_end_value", 0),
                "peak_time_s": row.get("peak_time_s", 0),
                "normalized_heat_flow_w_g_at_peak": row.get("peak_value", 0),
                "normalized_heat_j_g_at_peak": row.get("peak_j_g", 0),
            }
            results.append(result_data)

        return pd.DataFrame(results)

    def _merge_slope_results(
        self,
        max_slope_results: pd.DataFrame,
        mean_slope_results: pd.DataFrame,
        dormant_hf_results: pd.DataFrame,
        astm_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge max slope and mean slope results into comprehensive DataFrame."""
        if (
            max_slope_results.empty
            and mean_slope_results.empty
            and dormant_hf_results.empty
            and astm_results.empty
        ):
            return pd.DataFrame()

        # Use outer join to combine results by sample
        if max_slope_results.empty:
            return mean_slope_results
        if mean_slope_results.empty:
            return max_slope_results

        combined = pd.merge(
            max_slope_results,
            mean_slope_results,
            on=["sample", "sample_short"],
            how="outer",
            suffixes=("", "_duplicate"),
        )

        combined = pd.merge(
            combined,
            dormant_hf_results,
            on=["sample", "sample_short"],
            how="outer",
            suffixes=("", "_duplicate"),
        )

        combined = pd.merge(
            combined,
            astm_results,
            on=["sample", "sample_short"],
            how="outer",
            suffixes=("", "_duplicate"),
        )

        duplicate_cols = [col for col in combined.columns if col.endswith("_duplicate")]
        combined = combined.drop(columns=duplicate_cols)

        return combined

    def _plot_combined_slope_analysis(
        self,
        results: pd.DataFrame,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        plot_type: str,
        regex: Optional[str],
        plotpath: Optional[pathlib.Path],
        ax,
    ):
        """
        Plot combined slope analysis results based on plot_type parameter.

        Parameters
        ----------
        results : pd.DataFrame
            Combined results containing both max and mean slope data
        target_col : str
            Column name for heat flow data
        age_col : str
            Column name for time data
        plot_type : str
            Type of plot to show: 'max', 'mean', or 'both'
            - 'max': Shows only maximum slope analysis plot
            - 'mean': Shows only mean slope (flank tangent) analysis plot
            - 'both': Shows both analysis types (separate plots for each)
        regex : str, optional
            Regex to filter samples
        plotpath : pathlib.Path, optional
            Path to save plots
        cutoff_min : float, optional
            Cutoff time in minutes
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on
        """
        # Validate plot_type parameter
        valid_plot_types = ["max", "mean", "both"]
        cutoff_min = params.cutoff.cutoff_min

        if plot_type not in valid_plot_types:
            raise ValueError(
                f"plot_type must be one of {valid_plot_types}, got '{plot_type}'"
            )

        # For now, plot using the existing unified plotting approach
        # This could be enhanced to show both slope methods simultaneously
        for _, result_row in results.iterrows():
            sample = result_row["sample"]
            sample_short = result_row["sample_short"]

            # Get sample data
            sample_data = self._get_filtered_sample_data(
                sample, age_col, cutoff_time_min=cutoff_min
            )
            if sample_data.empty:
                continue

            if not pd.isna(
                result_row.onset_time_s_from_mean_slope or result_row.onset_time_s_from_max_slope
            ):
                self._plotter.plot_tangent_analysis(
                    sample_data,
                    sample_short,
                    params,
                    ax=ax,
                    age_col=age_col,
                    target_col=target_col,
                    cutoff_time_min=cutoff_min,
                    analysis_type=plot_type,  # Use correct analysis type
                    results=result_row.to_frame().T,
                    figsize=(7, 5),
                )
            self._save_and_show_plot(
                plotpath, f"{plot_type}_slope_{sample_short}.png", ax
            )


    # Backward compatibility methods
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
        """
        Get reaction onset via maximum slope intersection method.

        This is a wrapper around get_peak_onset_via_slope for backward compatibility.
        Returns only the max slope related columns for compatibility.
        """
        full_results = self.get_mainpeak_params(
            processparams=processparams,
            target_col=target_col,
            age_col=age_col,
            show_plot=show_plot,
            regex=regex,
            ax=ax,
            plot_type="max",

            #time_discarded_s=time_discarded_s,
            #intersection=intersection,
            #xunit=xunit,
        )

        if full_results.empty:
            return full_results

        # Extract only max slope related columns for backward compatibility
        # max_slope_cols = [
        #     col
        #     for col in full_results.columns
        #     if col.startswith("max_slope_") or col in ["sample", "sample_short"]
        # ]

        # result = full_results[max_slope_cols].copy()

        # Rename columns to match old API
        # column_mapping = {
        #     "onset_time_s_from_max_slope": "onset_time_s",
        #     "max_slope_onset_time_min": "onset_time_min",
        #     "max_slope_value": "maximum_slope",
        #     "max_slope_time_s": "maximum_slope_time_s",
        # }

        # for old_name, new_name in column_mapping.items():
        #     if old_name in result.columns:
        #         result = result.rename(columns={old_name: new_name})

        return full_results
    

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
        ax=None,
    ) -> pd.DataFrame:
        """
        Determine tangent to ascending flank of peak by averaging over sections.

        This is a wrapper around get_peak_onset_via_slope for backward compatibility.
        Returns only the mean slope related columns for compatibility.
        """
        full_results = self.get_peak_onset_via_slope(
            processparams=processparams,
            target_col=target_col,
            age_col=age_col,
            cutoff_min=cutoff_min,
            show_plot=show_plot,
            regex=regex,
            plotpath=plotpath,
            ax=ax,
            flank_fraction_start=flank_fraction_start,
            flank_fraction_end=flank_fraction_end,
            window_size=window_size,
        )

        if full_results.empty:
            return full_results

        # Extract only mean slope related columns for backward compatibility
        mean_slope_cols = [
            col
            for col in full_results.columns
            if col.startswith("mean_slope_")
            or col in ["sample", "sample_short", "peak_time_s", "peak_value"]
        ]

        result = full_results[mean_slope_cols].copy()

        # Rename columns to match old API
        column_mapping = {
            "mean_slope_onset_time_s": "x_intersection",
            "mean_slope_value": "tangent_slope",
            "mean_slope_time_s": "tangent_time_s",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in result.columns:
                result = result.rename(columns={old_name: new_name})

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
        dorm_hf = analyzer.get_dormant_period_heatflow(
            self._data, peaks, regex, upper_dormant_thresh_w_g
        )

        if not dorm_hf.empty:
            return dorm_hf
        else:
            return pd.DataFrame()

    def get_astm_c1679_characteristics(
        self,
        processparams: Optional[ProcessingParameters] = None,
        individual: bool = True,
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
        df = analyzer.get_astm_c1679_characteristics(
            self._data, peaks, individual, regex
        )
        return df

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

        analyzer = AverageSlopeAnalyzer(params)
        result = analyzer.get_average_slope(
            self._data, max_slopes, onsets, target_col, age_col, regex
        )
        return result

    def _plot_tangent_analysis_unified(
        self,
        results: pd.DataFrame,
        analysis_type: str,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        cutoff_time_min: Optional[float] = None,
        intersection: str = "dormant_hf",
        xunit: str = "s",
        time_discarded_s: float = 900,
        ax=None,
        # Additional data for onset intersection analysis
        max_slopes: Optional[pd.DataFrame] = None,
        dormant_hfs: Optional[pd.DataFrame] = None,
        onsets: Optional[pd.DataFrame] = None,
    ):
        """
        Unified plotting method for tangent-based analysis results.

        This method handles both flank tangent and onset intersection analysis,
        with the main difference being how the slope is determined:
        - Flank tangent: Uses averaged slope over a window
        - Max slope: Uses single point with maximum gradient

        Parameters
        ----------
        results : pd.DataFrame
            Results from the analysis (tangent results for flank, onsets for max slope)
        analysis_type : str
            Either 'flank_tangent' or 'max_slope_onset'
        target_col : str
            Column name for heat flow data
        age_col : str
            Column name for time data
        regex : str, optional
            Regex to filter samples
        plotpath : pathlib.Path, optional
            Path to save plots
        cutoff_time_min : float, optional
            Cutoff time in minutes
        intersection : str
            Type of intersection for onset analysis ('dormant_hf' or 'abscissa')
        xunit : str
            Time unit for plotting
        time_discarded_s : float
            Time to discard for onset analysis
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on
        max_slopes : pd.DataFrame, optional
            Required for onset intersection analysis
        dormant_hfs : pd.DataFrame, optional
            Required for onset intersection analysis with dormant_hf
        onsets : pd.DataFrame, optional
            Required for onset intersection analysis
        """
        try:
            if analysis_type == "flank_tangent":
                self._plot_flank_tangent_unified(
                    results, target_col, age_col, regex, plotpath, cutoff_time_min, ax
                )
            elif analysis_type == "max_slope_onset":
                self._plot_onset_intersection_unified(
                    results,
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
            else:
                raise ValueError(f"Unknown analysis_type: {analysis_type}")

        except Exception as e:
            logger.error(f"Error plotting tangent analysis results: {e}")
            print(f"Plotting failed: {e}")

    def _plot_flank_tangent_unified(
        self,
        results: pd.DataFrame,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        cutoff_time_min: Optional[float] = None,
        ax=None,
    ):
        """Plot flank tangent analysis results using unified SimplePlotter."""
        for _, result_row in results.iterrows():
            sample = result_row["sample"]
            sample_short = result_row["sample_short"]

            # Get sample data
            sample_data = self._get_filtered_sample_data(
                sample, age_col, cutoff_time_min=cutoff_time_min
            )
            if sample_data.empty:
                continue

            # Create a DataFrame with just this result for plotting
            single_result = pd.DataFrame([result_row])

            # Use unified plotting method
            self._plotter.plot_tangent_analysis(
                sample_data,
                sample_short,
                ax=ax,
                age_col=age_col,
                target_col=target_col,
                cutoff_time_min=cutoff_time_min,
                analysis_type="flank_tangent",
                tangent_results=single_result,
                figsize=(7, 5),
            )

            self._save_and_show_plot(plotpath, f"flank_tangent_{sample_short}.png", ax)

    def _plot_onset_intersection_unified(
        self,
        onsets: pd.DataFrame,
        max_slopes: Optional[pd.DataFrame],
        dormant_hfs: Optional[pd.DataFrame],
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        intersection: str = "dormant_hf",
        xunit: str = "s",
        time_discarded_s: float = 900,
        ax=None,
    ):
        """Plot onset intersection analysis results using unified SimplePlotter."""
        if max_slopes is None:
            raise ValueError("max_slopes required for onset intersection analysis")

        for _, onset_row in onsets.iterrows():
            sample = onset_row["sample"]

            # Get sample data
            sample_data = self._get_filtered_sample_data(
                sample, age_col, time_discarded_s=time_discarded_s
            )
            if sample_data.empty:
                continue

            # Use unified plotting method
            self._plotter.plot_tangent_analysis(
                sample_data,
                sample,
                ax=ax,
                age_col=age_col,
                target_col=target_col,
                analysis_type="onset_intersection",
                max_slopes=max_slopes,
                dormant_hfs=dormant_hfs,
                onsets=onsets,
                intersection=intersection,
                xunit=xunit,
                figsize=(12, 8),
            )

            # Note: plotpath not available in this context, only show plot
            self._save_and_show_plot(None, f"onset_intersection_{sample}.png", ax)

    def _get_filtered_sample_data(
        self,
        sample: str,
        age_col: str,
        cutoff_time_min: Optional[float] = None,
        time_discarded_s: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Get sample data with appropriate filtering applied.

        This consolidates the common data filtering logic used in both analysis types.
        """
        # Get sample data - handle both 'sample' and 'sample_short' columns
        sample_data = self._data[
            (self._data["sample"] == sample)
            | (self._data.get("sample_short", "") == sample)
        ]

        if sample_data.empty:
            return sample_data

        # Apply cutoff time filtering
        if cutoff_time_min is not None:
            cutoff_seconds = cutoff_time_min * 60
            sample_data = sample_data[sample_data[age_col] >= cutoff_seconds]

        # Apply time discarded filtering (for onset analysis)
        if time_discarded_s is not None and time_discarded_s > 0:
            sample_data = sample_data[sample_data[age_col] >= time_discarded_s]

        return sample_data

    def _save_and_show_plot(self, plotpath: Optional[pathlib.Path], filename: str, ax):
        """Handle plot saving and showing - common logic for both analysis types."""
        if plotpath:
            plot_file = plotpath / filename
            import matplotlib.pyplot as plt

            plt.savefig(plot_file, dpi=300, bbox_inches="tight")

        import matplotlib.pyplot as plt

        if not ax:
            plt.show()

    def _plot_flank_tangent_results(
        self,
        results: pd.DataFrame,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        cutoff_time_min: Optional[float] = None,
        ax=None,
    ):
        """
        Plot flank tangent analysis results using SimplePlotter.

        This is a wrapper around the unified plotting method for backward compatibility.
        """
        return self._plot_tangent_analysis_unified(
            results=results,
            analysis_type="flank_tangent",
            target_col=target_col,
            age_col=age_col,
            regex=regex,
            plotpath=plotpath,
            cutoff_time_min=cutoff_time_min,
            ax=ax,
        )

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
        """
        Plot onset intersection analysis results using SimplePlotter.

        This is a wrapper around the unified plotting method for backward compatibility.
        """
        return self._plot_tangent_analysis_unified(
            results=onsets,
            analysis_type="max_slope_onset",
            target_col=target_col,
            age_col=age_col,
            regex=regex,
            intersection=intersection,
            xunit=xunit,
            time_discarded_s=time_discarded_s,
            ax=ax,
            max_slopes=max_slopes,
            dormant_hfs=dormant_hfs,
            onsets=onsets,
        )

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
