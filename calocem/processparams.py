import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleParamRule:
    """
    Regex-based override rule for sample-specific processing parameters.

    Created with :meth:`ProcessingParameters.add_sample_param_rule`.

    Attributes
    ----------
    pattern : str
        Regular expression searched in the sample name.
    overrides : dict
        Nested dictionary of parameter groups and values to override, e.g.
        ``{"cutoff": {"cutoff_min": 30}}``.
    priority : int
        Rules are applied in ascending priority, so higher values win.
    """

    pattern: str
    overrides: dict[str, Any]
    priority: int = 100


@dataclass
class CutOffParameters:
    """
    Initial period excluded from the analysis.

    Attributes
    ----------
    cutoff_min : float
        Time in minutes; data before ``cutoff_min * 60`` seconds is ignored by
        the analyses. ``get_cumulated_heat_at_hours`` subtracts the heat released
        until then instead. Default 60.
    cutoff_max : int
        Reserved, currently not used.
    """

    cutoff_min: float = 60
    cutoff_max: int = 2880


@dataclass
class TianCorrectionParameters:
    """
    Time constants of the calorimeter for the Tian correction.

    Attributes
    ----------
    tau1 : float
        First time constant in seconds. Default 300.
    tau2 : float or None
        Second time constant in seconds. Set to None to correct with ``tau1``
        only. Default 100.
    """

    tau1: float = 300
    tau2: float | None = 100


@dataclass
class RollingMeanParameters:
    """
    Rolling mean applied to the heat flow before the gradient is taken.

    Used by the maximum slope analysis and ``plot_heatflow_with_gradient``.

    Attributes
    ----------
    apply : bool
        Apply the rolling mean. Default False.
    window : int
        Window size in data points. Default 11.
    """

    apply: bool = False
    window: int = 11


@dataclass
class MedianFilterParameters:
    """
    Median filter applied to the first and second derivative of the heat flow.

    Uses :func:`scipy.ndimage.median_filter`. The heat flow itself is not
    filtered.

    Attributes
    ----------
    apply : bool
        Apply the filter. Default False.
    size : int
        Filter size in data points. Default 7.
    """

    apply: bool = False
    size: int = 7


@dataclass
class NonLinSavGolParameters:
    """
    Reserved, currently not used.
    """

    apply: bool = False
    window: int = 11
    polynom: int = 3


@dataclass
class SplineInterpolationParameters:
    """
    Smoothing splines fitted to the first and second derivative of the heat flow.

    Uses :class:`scipy.interpolate.UnivariateSpline`.

    Attributes
    ----------
    apply : bool
        Apply the splines. Default False.
    smoothing_1st_deriv : float
        Smoothing factor ``s`` for the first derivative; larger is smoother.
        Default 1e-9.
    smoothing_2nd_deriv : float
        Smoothing factor ``s`` for the second derivative. Default 1e-9.
    """

    apply: bool = False
    smoothing_1st_deriv: float = 1e-9
    smoothing_2nd_deriv: float = 1e-9


@dataclass
class PeakDetectionParameters:
    """
    Settings of :func:`scipy.signal.find_peaks` for peak detection in the heat flow.

    Attributes
    ----------
    prominence : float
        Minimum peak prominence in W/g. Default 1e-5.
    distance : int
        Minimum distance between peaks in data points. Default 100.
    """

    prominence: float = 1e-5
    distance: int = 100


@dataclass
class GradientPeakDetectionParameters:
    """
    Reserved, currently not used.
    """

    prominence: float = 1e-9
    distance: int = 100
    width: int = 20
    rel_height: float = 0.05
    height: float = 1e-9
    use_first: bool = False
    use_largest_width: bool = False
    use_largest_width_height: bool = False


@dataclass
class DownSamplingParameters:
    """
    Adaptive downsampling applied when a Measurement is created.

    More points are kept where the heat flow curve bends, fewer in flat
    regions. Values are picked from the measured data, not interpolated.

    Attributes
    ----------
    apply : bool
        Apply downsampling. Default False.
    num_points : int
        Target number of points per sample (upper bound). Default 1000.
    smoothing_factor : float
        Relative smoothing of the spline used to estimate the curvature; scaled
        internally by the number of points and the variance of the heat flow.
        Default 1e-6.
    baseline_weight : float
        Weight of evenly spread points, so flat regions keep some points.
        Default 0.1.
    section_split : bool
        Downsample the data before and after ``section_split_time_s``
        separately, e.g. to keep an early narrow peak. Default False.
    section_split_time_s : int
        Split time in seconds. Default 1000.
    """
    apply: bool = False
    num_points: int = 1000
    smoothing_factor: float = 1e-6
    baseline_weight: float = 0.1
    section_split: bool = False
    section_split_time_s: int = 1000


@dataclass
class PreProcessParameters:
    """
    Reserved, currently not used.
    """

    infer_heat: bool = False


@dataclass
class SlopeAnalysisParameters:
    """
    Settings of the slope analyses of the main hydration peak.

    Attributes
    ----------
    flank_fraction_start : float
        Start of the ascending flank as fraction of the height between the
        minimum before the peak and the peak. Default 0.35.
    flank_fraction_end : float
        End of the ascending flank, same scale. Default 0.55.
    window_size : float
        Width of the averaging windows as fraction of the time span of the flank.
        Default 0.1.
    first_ascending_fraction_of_max : float
        ``method="ascending"``: threshold as fraction of the maximum heat flow
        after the cutoff, in (0, 1]. Default 0.2.
    first_ascending_range_method : str
        ``method="ascending"``: "fraction" (default) ends the segment at
        ``first_ascending_fraction_of_max * max``; "delta" ends it at the minimum
        plus ``first_ascending_delta_y_w_g``.
    first_ascending_delta_y_w_g : float
        Heat flow rise in W/g for the "delta" method. Default 0.001.
    flexible : float
        "delta" method: increases the delta for high peaks; 0 (default) disables
        the adaptation.
    """

    flank_fraction_start: float = 0.35
    flank_fraction_end: float = 0.55
    window_size: float = 0.1
    first_ascending_fraction_of_max: float = 0.2
    first_ascending_range_method: str = "fraction"
    first_ascending_delta_y_w_g: float = 0.001
    flexible: float = 0.0


@dataclass
class DeconvolutionParameters:
    """
    Settings of the peak deconvolution fit.

    Attributes
    ----------
    n_peaks : int
        Number of fitted peaks. Default 2.
    peak_shape : str
        Currently ignored; select the shape with the ``peak_shape`` argument of
        ``Measurement.get_deconvolution``.
    baseline_mode : str
        "constant" (default), "linear", "chebyshev" or "none".
    chebyshev_degree : int
        Degree of the Chebyshev baseline. Default 2.
    max_nfev : int
        Maximum number of optimizer iterations. Default 5000.
    max_fit_points : int
        The data is subsampled to at most this many points. Default 2500.
    min_points : int
        Samples with fewer points after the cutoff are skipped. Default 30.
    min_peak_distance_fraction : float
        Minimum distance of the initial peak guesses as fraction of the number of
        points. Default 0.05.
    min_sigma_fraction, max_sigma_fraction : float
        Width bounds of Gaussian peaks as fraction of the time range. Defaults
        0.005 and 0.3.
    min_peak_time_separation_fraction : float
        Minimum separation of the fitted peak times as fraction of the time
        range. Default 1e-4.
    relative_intensity_upper_bounds : list of float or None
        One value per peak in (0, 1]: maximum share of the total amplitude.
    peak_width_upper_bounds : list of float or None
        One value per peak: maximum width.
    """

    n_peaks: int = 2
    peak_shape: str = "lognormal"
    baseline_mode: str = "constant"
    chebyshev_degree: int = 2
    max_nfev: int = 5000
    max_fit_points: int = 2500
    min_points: int = 30
    min_peak_distance_fraction: float = 0.05
    min_sigma_fraction: float = 0.005
    max_sigma_fraction: float = 0.30
    min_peak_time_separation_fraction: float = 1e-4
    relative_intensity_upper_bounds: list[float] | None = None
    peak_width_upper_bounds: list[float] | None = None


@dataclass
class PlottingParameters:
    """
    Appearance of the analysis plots of ``get_mainpeak_params``.

    ``Measurement.plot`` does not use these settings; ``plot_heatflow_with_gradient``
    uses ``figsize`` only.

    Attributes
    ----------
    figsize : tuple of float
        Figure size in inches, used when no axes is passed. Default (10, 6).
    time_unit : str
        "seconds" (default), "minutes" or "hours".
    heat_unit : str
        "W" (default) or "mW".
    show_plot_title : bool
        Show a title. Default True.
    legend_pos : str
        "best" (default) or "outside" (right of the plot).
    plot_title : str or list of str
        With metadata loaded: metadata columns whose values form the title, e.g.
        ``["cement_name", "w_c"]``. Default "filename".
    xlims, ylims : tuple of float or None
        Axis limits in plot units. ``ylims`` is only applied together with
        ``xlims``. Default None.
    """
    figsize: tuple[float, float] = (10, 6)
    time_unit: str = "seconds"
    heat_unit: str = "W"
    show_plot_title: bool = True
    legend_pos: str = "best"
    plot_title: str | list[str] = "filename"
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None

@dataclass
class ProcessingParameters:
    """
    All processing parameters of CaloCem, grouped by topic.

    Create an instance, change the values you need and pass it to the methods
    of :class:`calocem.Measurement`. See the Processing Parameters guide for
    which analyses use which settings.

    Attributes
    ----------
    cutoff : CutOffParameters
        Initial period excluded from the analysis.
    time_constants : TianCorrectionParameters
        Time constants for the Tian correction.
    peakdetection : PeakDetectionParameters
        Peak detection in the heat flow.
    gradient_peakdetection : GradientPeakDetectionParameters
        Reserved, currently not used.
    rolling_mean : RollingMeanParameters
        Rolling mean of the heat flow before differentiation.
    median_filter : MedianFilterParameters
        Median filter of the derivatives.
    nonlin_savgol : NonLinSavGolParameters
        Reserved, currently not used.
    spline_interpolation : SplineInterpolationParameters
        Smoothing splines of the derivatives.
    downsample : DownSamplingParameters
        Adaptive downsampling while loading.
    preprocess : PreProcessParameters
        Reserved, currently not used.
    slope_analysis : SlopeAnalysisParameters
        Slope analyses of the main hydration peak.
    deconvolution : DeconvolutionParameters
        Peak deconvolution.
    plotting : PlottingParameters
        Appearance of the analysis plots.
    sample_param_rules : list of SampleParamRule
        Sample-specific overrides, see :meth:`add_sample_param_rule`.

    Examples
    --------
    >>> processparams = ProcessingParameters()
    >>> processparams.cutoff.cutoff_min = 30
    >>> processparams.spline_interpolation.apply = True
    """

    cutoff: CutOffParameters = field(default_factory=CutOffParameters)
    time_constants: TianCorrectionParameters = field(
        default_factory=TianCorrectionParameters
    )

    # peak detection params
    peakdetection: PeakDetectionParameters = field(
        default_factory=PeakDetectionParameters
    )
    gradient_peakdetection: GradientPeakDetectionParameters = field(
        default_factory=GradientPeakDetectionParameters
    )

    # smoothing params
    rolling_mean: RollingMeanParameters = field(default_factory=RollingMeanParameters)
    median_filter: MedianFilterParameters = field(
        default_factory=MedianFilterParameters
    )
    nonlin_savgol: NonLinSavGolParameters = field(
        default_factory=NonLinSavGolParameters
    )
    spline_interpolation: SplineInterpolationParameters = field(
        default_factory=SplineInterpolationParameters
    )
    # preprocessing params
    downsample: DownSamplingParameters = field(default_factory=DownSamplingParameters)
    preprocess: PreProcessParameters = field(default_factory=PreProcessParameters)
    # slope analysis params
    slope_analysis: SlopeAnalysisParameters = field(
        default_factory=SlopeAnalysisParameters
    )
    deconvolution: DeconvolutionParameters = field(
        default_factory=DeconvolutionParameters
    )
    plotting: PlottingParameters = field(default_factory=PlottingParameters)
    sample_param_rules: list[SampleParamRule] = field(default_factory=list)

    def add_sample_param_rule(
        self,
        pattern: str,
        overrides: dict[str, Any],
        priority: int = 100,
    ) -> None:
        """Add a regex rule to override parameters for matching sample names.

        Note
        ----
        Rules are currently only honoured by the first-ascending-slope analysis,
        i.e. ``get_mainpeak_params(method="ascending")``. All other analyses use
        the base parameters for every sample.

        Examples
        --------
        >>> processparams.add_sample_param_rule(
        ...     r"^c3a", {"slope_analysis": {"first_ascending_range_method": "delta"}}
        ... )
        """
        self.sample_param_rules.append(
            SampleParamRule(pattern=pattern, overrides=overrides, priority=priority)
        )

    def resolve_for_sample(self, sample_name: str) -> "ProcessingParameters":
        """Resolve effective parameters for a sample based on regex override rules."""
        if not self.sample_param_rules:
            return self

        matched_rules = sorted(
            (rule for rule in self.sample_param_rules if re.search(rule.pattern, sample_name)),
            key=lambda item: item.priority,
        )

        if not matched_rules:
            return self

        resolved = deepcopy(self)
        for rule in matched_rules:
            self._apply_nested_overrides(resolved, rule.overrides)

        return resolved

    def _apply_nested_overrides(self, target: Any, overrides: dict[str, Any]) -> None:
        """Apply nested dictionary overrides to a dataclass-like object."""
        for key, value in overrides.items():
            if not hasattr(target, key):
                raise AttributeError(f"Unknown processing parameter field: {key}")

            current_value = getattr(target, key)
            if isinstance(value, dict):
                self._apply_nested_overrides(current_value, value)
            else:
                setattr(target, key, value)
