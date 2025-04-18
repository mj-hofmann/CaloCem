
from dataclasses import dataclass, field

@dataclass
class CutOffParameters:
    cutoff_min: int = 30
    cutoff_max: int = 2880 # 2 days in minutes


@dataclass
class TianCorrectionParameters:
    """
    Parameters related to time constants used in Tian's correction method for thermal analysis. The default values are defined in the TianCorrectionParameters class.

    Parameters
    ----------
    tau1 : int
        Time constant for the first correction step in Tian's method. The default value is 300.

    tau2 : int
        Time constant for the second correction step in Tian's method. The default value is 100.
    """

    tau1: int = 300
    tau2: int = 100


@dataclass
class RollingMeanParameters:
    apply: bool = False
    window: int = 11


@dataclass
class MedianFilterParameters:
    """Parameters for the application of a Median filter to the data. The SciPy method `median_filter` is applied. [Link to method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html)


    Parameters
    ----------
    apply: bool
        default is false. If `True`, a median filter is applied. The Scipy function `median_filter` is  applied.   
    size: int
        The size of the median filter (see the SciPy documentation)
    
    """
    apply: bool = False
    size: int = 7


@dataclass
class NonLinSavGolParameters:
    apply: bool = False
    window: int = 11
    polynom: int = 3


@dataclass
class SplineInterpolationParameters:
    """Parameters for spline interpolation of heat flow data.

    Parameters
    ----------

    apply :
        Flag indicating whether spline interpolation should be applied to the heat flow data. The default value is False.

    smoothing_1st_deriv :
        Smoothing parameter for the first derivative of the heat flow data. The default value is 1e-9.

    smoothing_2nd_deriv :
        Smoothing parameter for the second derivative of the heat flow data. The default value is 1e-9.

    """

    apply: bool = False
    smoothing_1st_deriv: float = 1e-9
    smoothing_2nd_deriv: float = 1e-9


@dataclass
class PeakDetectionParameters:
    """
    Parameters that control the identication of peaks during peak detection. 

    Parameters
    ----------

    prominence: float
        The minimum prominence of the peak
    distance: int
        The minimum distance of the peak.
    """
    prominence: float = 1e-5
    distance: int = 100


@dataclass
class GradientPeakDetectionParameters:
    """
    Parameters that control the identifcation of Peaks in the first derivative (gradient) of the heat flow data. Under the hood the SciPy `find_peaks()` is used [Link to SciPy method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). 

    Parameters
    ----------
    prominence: float
        Minimum prominence of the peak
    distance: int
        Minimum distance
    width: int
        Minimum width
    rel_height: float
        Relative height of the peak
    height: float
        Minimum height of the peak
    use_first: bool
        If true, only the first peak will be used
    use_largest_width: bool
        If true the peak with the largest peak width will be used.
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
    apply: bool = False
    num_points: int = 1000
    smoothing_factor: float = 1e-10
    baseline_weight: float = 0.1
    section_split: bool = False
    section_split_time_s: int = 1000

@dataclass
class PreProcessParameters:
    """
    Parameters for preprocessing the data before analysis.

    Attributes
    ----------
    infer_heat: bool
        If True, the heat flow is inferred from the data. This is useful for data that does not have a heat flow column.
    """

    infer_heat: bool = False
@dataclass
class ProcessingParameters:
    """
    A data class for storing all processing parameters for calorimetry data.

    This class aggregates various processing parameters, including cutoff criteria, time constants for the Tian correction, and parameters for peak detection and gradient peak detection.

    Attributes
    ----------

    cutoff :
        Parameters defining the cutoff criteria for the analysis.
        Currently only cutoff_min is implemented, which defines the minimum time in minutes for the analysis. The default value is defined in the CutOffParameters class.

    time_constants : TianCorrectionParameters
        Parameters related to time constants used in Tian's correction method for thermal analysis. he default values are defined in the
        TianCorrectionParameters class.

    peakdetection : PeakDetectionParameters
        Parameters for detecting peaks in the thermal analysis data. This includes settings such as the minimum
        prominence and distance between peaks. The default values are defined in the PeakDetectionParameters class.

    gradient_peakdetection : GradientPeakDetectionParameters
        Parameters for detecting peaks based on the gradient of the thermal analysis data. This includes more
        nuanced settings such as prominence, distance, width, relative height, and the criteria for selecting peaks
        (e.g., first peak, largest width). The default values are defined in the GradientPeakDetectionParameters class.

    downsample : DownSamplingParameters
        Parameters for adaptive downsampling of the thermal analysis data. This includes settings such as the number of points,
        smoothing factor, and baseline weight. The default values are defined in the DownSamplingParameters class.

    spline_interpolation: SplineInterpolationParameters
        Parameters which control the interpolation of the first and second derivative of the data. If no smoothing is applied the derivatives often become very noisy.

    Examples
    --------

    Define a set of processing parameters for thermal analysis data.

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
    preprocess: PreProcessParameters = field(
        default_factory=PreProcessParameters
    )
