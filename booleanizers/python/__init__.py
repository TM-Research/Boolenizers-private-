"""Thermometer encoders for TM experiments (unsupervised ``fit(X)`` in ``codes/`` benchmarks)."""

from .base import ThermometerEncoder
from .nte_uniform import NTEUniform
from .nte_batch_quantile import NTEBatchQuantile
from .adaptive_gaussian import AdaptiveGaussian
from .single_speed_p2 import SingleSpeedP2
from .sketch_gk import SketchGK
from .sketch_tdigest import SketchTDigest
from .twine import TWINE
from .twine_v2 import TWINEv2
from .twine_v3 import TWINEv3
from .twine_lite import TWINELite
from .twine_streaming_values import TWINEStreamingValues
from .twine_stream_trade import TWINEStreamTrade
from .standard_wrapper import StandardBinarizerWrapper
from .standard_binarizer_native import StandardBinarizerNative
from .acfb import ACFB
from .ssl import SSL
from .sdqb import SDQB
from .qb_espresso import QBEspresso
from .dpb import DynamicPulseBinarizer
from .ssb import SpectralStabilityBinarizer
from .amb import AdaptiveMomentumBinarizer
from .kmb import KnownMethodsBinarizer
from .kfb import KalmanFilterBinarizer
from .dtb import DecisionTreeBinarizer
from .sqf import SignalQuantileFusion
from .drb import DriftRobustBinarizer
from .aqb import AdaptiveQuantileBinarizer
from .rgb import ResonantGradientBinarizer
from .mwb import MovingWindowBinarizer
from .ddb import DualDynamicsBinarizer
from .rgb2 import ResonantGradientBinarizerV2
from .mwb2 import MovingWindowBinarizerV2
from .prb import PulseResonanceBinarizer
from .sgb import SignalGradientBinarizer
from .obb import OnlineBollingerBinarizer
from .oatb import OnlineATRBinarizer
from .ormb import OnlineRSIMACDBinarizer
from .odmb import OnlineDeltaMomentumBinarizer
from .oqtb import OnlineQuantileTrackerBinarizer
from .oub import OnlineUniversalBinarizer
from .oqsb import OnlineQuantileSignalBinarizer
from .ogb import OnlineGeneralizedBinarizer
from .ogb_fast import OGBFast
from .kbins_thermometer import KBinsThermometer
from .glade import GLADEEncoder, GLADEBooleanizer

__all__ = [
    "ThermometerEncoder",
    "NTEUniform",
    "NTEBatchQuantile",
    "AdaptiveGaussian",
    "SingleSpeedP2",
    "SketchGK",
    "SketchTDigest",
    "TWINE",
    "TWINEv2",
    "TWINEv3",
    "TWINELite",
    "StandardBinarizerWrapper",
    "StandardBinarizerNative",
    "ACFB",
    "SSL",
    "SDQB",
    "QBEspresso",
    "DynamicPulseBinarizer",
    "SpectralStabilityBinarizer",
    "AdaptiveMomentumBinarizer",
    "KnownMethodsBinarizer",
    "KalmanFilterBinarizer",
    "DecisionTreeBinarizer",
    "SignalQuantileFusion",
    "DriftRobustBinarizer",
    "AdaptiveQuantileBinarizer",
    "ResonantGradientBinarizer",
    "MovingWindowBinarizer",
    "DualDynamicsBinarizer",
    "ResonantGradientBinarizerV2",
    "MovingWindowBinarizerV2",
    "PulseResonanceBinarizer",
    "SignalGradientBinarizer",
    "OnlineBollingerBinarizer",
    "OnlineATRBinarizer",
    "OnlineRSIMACDBinarizer",
    "OnlineDeltaMomentumBinarizer",
    "OnlineQuantileTrackerBinarizer",
    "OnlineUniversalBinarizer",
    "OnlineQuantileSignalBinarizer",
    "OnlineGeneralizedBinarizer",
    "OGBFast",
    "KBinsThermometer",
    "GLADEEncoder",
    "GLADEBooleanizer",
]
