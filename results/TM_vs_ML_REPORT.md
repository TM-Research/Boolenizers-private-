# Tsetlin Machine + ALL booleanizers vs ML — 148 datasets

Per dataset: every booleanizer in the library binarizes the same preprocessed features; the best feeds a per-dataset-configured DeterministicTM. Compared to best of XGBoost/LightGBM/RandomForest/ExtraTrees on the identical split. Metric = macro-F1.

- **TM ≥ ML on 101/148 datasets** (68%).
- Mean macro-F1: **TM 0.7895** vs **ML 0.7840** (mean Δ +0.0055).
- Booleanizers that won at least once (40 distinct): GLADEEncoder×13, GLADEBooleanizer×11, OGBFast×10, OnlineQuantileTrackerBinarizer×6, OnlineGeneralizedBinarizer×6, ACFB×6, NTEUniform×6, SDQB×6, MovingWindowBinarizerV2×5, ResonantGradientBinarizerV2×5, PulseResonanceBinarizer×5, DriftRobustBinarizer×4.

## Per-booleanizer mean TM macro-F1 (across datasets) — top 15

| Booleanizer | mean F1 | #datasets | #wins |
|---|--:|--:|--:|
| GLADEEncoder | 0.7510 | 148 | 13 |
| OnlineGeneralizedBinarizer | 0.7498 | 148 | 6 |
| KnownMethodsBinarizer | 0.7491 | 148 | 2 |
| ResonantGradientBinarizerV2 | 0.7489 | 148 | 5 |
| AdaptiveQuantileBinarizer | 0.7480 | 148 | 1 |
| StandardBinarizerNative | 0.7472 | 148 | 4 |
| QBEspresso | 0.7468 | 147 | 4 |
| KalmanFilterBinarizer | 0.7468 | 148 | 1 |
| SDQB | 0.7466 | 148 | 6 |
| AdaptiveMomentumBinarizer | 0.7465 | 148 | 2 |
| PulseResonanceBinarizer | 0.7463 | 148 | 5 |
| GLADEBooleanizer | 0.7461 | 148 | 11 |
| MovingWindowBinarizer | 0.7458 | 148 | 2 |
| NTEBatchQuantile | 0.7455 | 148 | 3 |
| SingleSpeedP2 | 0.7448 | 148 | 3 |

## A. UCR time-series archive: 116 datasets — TM≥ML 84/116, mean TM 0.777 vs ML 0.768

## B. Original 12 time-series: 12 datasets — TM≥ML 10/12, mean TM 0.942 vs ML 0.930

## C. Cyber / IDS multi-class: 20 datasets — TM≥ML 7/20, mean TM 0.772 vs ML 0.788

## Full results (sorted by TM macro-F1)

| Dataset | kind | C | TM best booleanizer | TM F1 | ML best | ML F1 | Δ | winner |
|---|---|--:|---|--:|---|--:|--:|---|
| Plane | time | 7 | QBEspresso | 1.000 | LightGBM | 0.984 | +0.016 | TM |
| SonyAIBORobotSurface1 | time | 2 | GLADEBooleanizer | 1.000 | ExtraTrees | 0.995 | +0.005 | TM |
| Trace | time | 4 | OnlineGeneralizedBinarizer | 1.000 | ExtraTrees | 0.967 | +0.033 | TM |
| 5gcid-multiclass | tabu | 4 | OGBFast | 1.000 | XGBoost | 1.000 | +0.000 | TM |
| domain-info-2024-multiclass | tabu | 3 | GLADEBooleanizer | 1.000 | XGBoost | 1.000 | +0.000 | TM |
| BME | time | 3 | OGBFast | 1.000 | RandomForest | 1.000 | +0.000 | TM |
| CBF | time | 3 | NTEBatchQuantile | 1.000 | RandomForest | 1.000 | +0.000 | TM |
| Chinatown | time | 2 | GLADEBooleanizer | 1.000 | LightGBM | 1.000 | +0.000 | TM |
| CinCECGTorso | time | 4 | SDQB | 1.000 | LightGBM | 0.998 | +0.002 | TM |
| Coffee | time | 2 | AdaptiveMomentumBinarizer | 1.000 | XGBoost | 0.941 | +0.059 | TM |
| DiatomSizeReduction | time | 4 | SDQB | 1.000 | RandomForest | 1.000 | +0.000 | TM |
| DodgerLoopWeekend | time | 2 | OGBFast | 1.000 | XGBoost | 1.000 | +0.000 | TM |
| ECGFiveDays | time | 2 | OGBFast | 1.000 | LightGBM | 0.996 | +0.004 | TM |
| FaceFour | time | 4 | MovingWindowBinarizerV2 | 1.000 | XGBoost | 0.971 | +0.029 | TM |
| FreezerSmallTrain | time | 2 | OGBFast | 1.000 | LightGBM | 1.000 | +0.000 | TM |
| Fungi | time | 18 | GLADEEncoder | 1.000 | RandomForest | 1.000 | +0.000 | TM |
| GunPointMaleVersusFemale | time | 2 | ACFB | 1.000 | ExtraTrees | 0.993 | +0.007 | TM |
| Meat | time | 3 | GLADEBooleanizer | 1.000 | LightGBM | 1.000 | +0.000 | TM |
| OliveOil | time | 4 | SignalQuantileFusion | 1.000 | LightGBM | 1.000 | +0.000 | TM |
| UMD | time | 3 | OGBFast | 1.000 | ExtraTrees | 1.000 | +0.000 | TM |
| FreezerRegularTrain | time | 2 | OGBFast | 0.999 | RandomForest | 1.000 | -0.001 | ML |
| Wafer | time | 2 | GLADEBooleanizer | 0.999 | XGBoost | 0.998 | +0.001 | TM |
| Mallat | time | 8 | DualDynamicsBinarizer | 0.999 | RandomForest | 0.996 | +0.003 | TM |
| TwoLeadECG | time | 2 | GLADEEncoder | 0.997 | LightGBM | 0.997 | +0.000 | TM |
| smart-digital | tabu | 10 | KBinsThermometer | 0.995 | RandomForest | 0.998 | -0.003 | ML |
| sd-iot | tabu | 10 | KalmanFilterBinarizer | 0.995 | XGBoost | 0.999 | -0.004 | ML |
| ChlorineConcentration | time | 3 | StandardBinarizerNative | 0.994 | LightGBM | 0.996 | -0.001 | ML |
| cicmaldroid-2020-multiclass | tabu | 5 | OnlineRSIMACDBinarizer | 0.990 | LightGBM | 0.951 | +0.039 | TM |
| SmoothSubspace | time | 3 | OGBFast | 0.989 | ExtraTrees | 0.989 | +0.000 | TM |
| GunPointAgeSpan | time | 2 | OnlineGeneralizedBinarizer | 0.985 | XGBoost | 0.963 | +0.022 | TM |
| cic-iov-2024-multiclass | tabu | 6 | TWINEv2 | 0.979 | XGBoost | 0.975 | +0.004 | TM |
| Strawberry | time | 2 | OGBFast | 0.978 | LightGBM | 0.978 | +0.000 | TM |
| SyntheticControl | time | 6 | GLADEEncoder | 0.978 | RandomForest | 0.994 | -0.016 | ML |
| UWaveGestureLibraryAll | time | 8 | DecisionTreeBinarizer | 0.978 | ExtraTrees | 0.972 | +0.006 | TM |
| Symbols | time | 6 | DecisionTreeBinarizer | 0.977 | ExtraTrees | 0.977 | -0.000 | ML |
| TwoPatterns | time | 4 | NTEUniform | 0.977 | ExtraTrees | 0.991 | -0.014 | ML |
| ItalyPowerDemand | time | 2 | OnlineATRBinarizer | 0.973 | RandomForest | 0.970 | +0.003 | TM |
| PowerCons | time | 2 | OnlineQuantileTrackerBinarizer | 0.972 | XGBoost | 0.963 | +0.009 | TM |
| SonyAIBORobotSurface2 | time | 2 | SingleSpeedP2 | 0.972 | LightGBM | 0.961 | +0.011 | TM |
| Fish | time | 7 | MovingWindowBinarizer | 0.971 | ExtraTrees | 0.953 | +0.019 | TM |
| FaceAll | time | 14 | DecisionTreeBinarizer | 0.969 | ExtraTrees | 0.946 | +0.023 | TM |
| MoteStrain | time | 2 | DriftRobustBinarizer | 0.968 | ExtraTrees | 0.971 | -0.003 | ML |
| GunPoint | time | 2 | NTEBatchQuantile | 0.967 | RandomForest | 0.967 | +0.000 | TM |
| GunPointOldVersusYoung | time | 2 | MovingWindowBinarizerV2 | 0.963 | ExtraTrees | 0.971 | -0.007 | ML |
| ddos-tnsm | tabu | 7 | AdaptiveMomentumBinarizer | 0.963 | LightGBM | 0.965 | -0.002 | ML |
| FacesUCR | time | 14 | KBinsThermometer | 0.962 | ExtraTrees | 0.943 | +0.019 | TM |
| MixedShapesSmallTrain | time | 5 | TWINEv3 | 0.959 | XGBoost | 0.952 | +0.007 | TM |
| StarLightCurves | time | 3 | TWINELite | 0.957 | ExtraTrees | 0.960 | -0.003 | ML |
| Ham | time | 2 | SDQB | 0.954 | LightGBM | 0.922 | +0.031 | TM |
| MixedShapesRegularTrain | time | 5 | TWINEv2 | 0.950 | LightGBM | 0.937 | +0.013 | TM |
| cybersoceval-hybrid-analysis-family | tabu | 5 | DecisionTreeBinarizer | 0.948 | ExtraTrees | 0.857 | +0.091 | TM |
| Wine | time | 2 | GLADEEncoder | 0.941 | LightGBM | 0.941 | -0.000 | ML |
| ArrowHead | time | 3 | ACFB | 0.939 | RandomForest | 0.908 | +0.031 | TM |
| NonInvasiveFetalECGThorax2 | time | 42 | QBEspresso | 0.936 | ExtraTrees | 0.920 | +0.016 | TM |
| Yoga | time | 2 | OnlineGeneralizedBinarizer | 0.934 | ExtraTrees | 0.946 | -0.012 | ML |
| NonInvasiveFetalECGThorax1 | time | 42 | PulseResonanceBinarizer | 0.927 | ExtraTrees | 0.894 | +0.032 | TM |
| SwedishLeaf | time | 15 | NTEUniform | 0.923 | ExtraTrees | 0.905 | +0.018 | TM |
| ECG200 | time | 2 | MovingWindowBinarizerV2 | 0.923 | ExtraTrees | 0.870 | +0.053 | TM |
| Rock | time | 4 | GLADEBooleanizer | 0.922 | LightGBM | 0.922 | +0.000 | TM |
| HandOutlines | time | 2 | DriftRobustBinarizer | 0.920 | RandomForest | 0.918 | +0.002 | TM |
| BeetleFly | time | 2 | MWAB | 0.916 | XGBoost | 0.916 | +0.000 | TM |
| BirdChicken | time | 2 | SDQB | 0.916 | XGBoost | 1.000 | -0.084 | ML |
| HouseTwenty | time | 2 | OnlineATRBinarizer | 0.915 | ExtraTrees | 0.871 | +0.044 | TM |
| MelbournePedestrian | time | 10 | GLADEBooleanizer | 0.910 | ExtraTrees | 0.900 | +0.010 | TM |
| edge-iiotset-multiclass | tabu | 15 | OnlineGeneralizedBinarizer | 0.908 | XGBoost | 0.974 | -0.065 | ML |
| DodgerLoopGame | time | 2 | TWINEv2 | 0.885 | ExtraTrees | 0.886 | -0.001 | ML |
| InsectEPGSmallTrain | time | 3 | TWINEv3 | 0.881 | ExtraTrees | 0.840 | +0.041 | TM |
| GesturePebbleZ2 | time | 6 | GLADEEncoder | 0.871 | ExtraTrees | 0.845 | +0.026 | TM |
| nids-bench-2026 | tabu | 13 | AQB | 0.869 | XGBoost | 0.867 | +0.001 | TM |
| DistalPhalanxOutlineCorrect | time | 2 | SketchTDigest | 0.856 | LightGBM | 0.849 | +0.008 | TM |
| ProximalPhalanxOutlineCorrect | time | 2 | SAQT | 0.851 | XGBoost | 0.841 | +0.010 | TM |
| DistalPhalanxOutlineAgeGroup | time | 3 | ResonantGradientBinarizerV2 | 0.842 | ExtraTrees | 0.844 | -0.002 | ML |
| SemgHandGenderCh2 | time | 2 | TWINEv2 | 0.839 | LightGBM | 0.807 | +0.032 | TM |
| ToeSegmentation1 | time | 2 | OGBFast | 0.839 | ExtraTrees | 0.797 | +0.042 | TM |
| Car | time | 4 | ResonantGradientBinarizer | 0.833 | ExtraTrees | 0.808 | +0.025 | TM |
| UWaveGestureLibraryX | time | 8 | SketchTDigest | 0.831 | ExtraTrees | 0.815 | +0.016 | TM |
| ACSF1 | time | 10 | GLADEEncoder | 0.831 | ExtraTrees | 0.803 | +0.027 | TM |
| ProximalPhalanxOutlineAgeGroup | time | 3 | ResonantGradientBinarizer | 0.829 | ExtraTrees | 0.812 | +0.017 | TM |
| anf-iot | tabu | 3 | GLADEEncoder | 0.824 | XGBoost | 0.843 | -0.019 | ML |
| PhalangesOutlinesCorrect | time | 2 | GLADEEncoder | 0.822 | RandomForest | 0.819 | +0.003 | TM |
| ToeSegmentation2 | time | 2 | OnlineBollingerBinarizer | 0.812 | LightGBM | 0.767 | +0.046 | TM |
| FordA | time | 2 | OnlineQuantileTrackerBinarizer | 0.812 | LightGBM | 0.823 | -0.011 | ML |
| MiddlePhalanxOutlineCorrect | time | 2 | GLADEBooleanizer | 0.812 | XGBoost | 0.796 | +0.016 | TM |
| ornl-msu | tabu | 37 | StandardBinarizerNative | 0.808 | ExtraTrees | 0.896 | -0.088 | ML |
| SemgHandSubjectCh2 | time | 5 | OnlineBollingerBinarizer | 0.797 | LightGBM | 0.707 | +0.091 | TM |
| GesturePebbleZ1 | time | 6 | PulseResonanceBinarizer | 0.792 | ExtraTrees | 0.792 | +0.000 | TM |
| FordB | time | 2 | SAQT | 0.788 | XGBoost | 0.788 | -0.000 | ML |
| MedicalImages | time | 10 | MovingWindowBinarizer | 0.773 | ExtraTrees | 0.785 | -0.011 | ML |
| UWaveGestureLibraryY | time | 8 | NTEUniform | 0.766 | RandomForest | 0.760 | +0.006 | TM |
| Adiac | time | 37 | GLADEEncoder | 0.765 | RandomForest | 0.702 | +0.064 | TM |
| UWaveGestureLibraryZ | time | 8 | ACFB | 0.762 | ExtraTrees | 0.760 | +0.002 | TM |
| InsectEPGRegularTrain | time | 3 | TWINELite | 0.760 | ExtraTrees | 0.734 | +0.026 | TM |
| Crop | time | 24 | QBEspresso | 0.756 | XGBoost | 0.742 | +0.014 | TM |
| Beef | time | 5 | ResonantGradientBinarizerV2 | 0.743 | LightGBM | 0.640 | +0.103 | TM |
| ElectricDevices | time | 7 | StandardBinarizerNative | 0.737 | XGBoost | 0.769 | -0.033 | ML |
| Lightning7 | time | 7 | SDQB | 0.731 | RandomForest | 0.667 | +0.064 | TM |
| SmallKitchenAppliances | time | 3 | ResonantGradientBinarizerV2 | 0.728 | ExtraTrees | 0.723 | +0.005 | TM |
| EOGHorizontalSignal | time | 12 | ACFB | 0.723 | ExtraTrees | 0.687 | +0.036 | TM |
| ShapesAll | time | 60 | StandardBinarizerNative | 0.721 | ExtraTrees | 0.768 | -0.047 | ML |
| Lightning2 | time | 2 | DriftRobustBinarizer | 0.720 | XGBoost | 0.745 | -0.025 | ML |
| CricketX | time | 12 | NTEBatchQuantile | 0.719 | ExtraTrees | 0.694 | +0.025 | TM |
| InsectWingbeatSound | time | 11 | GLADEEncoder | 0.708 | LightGBM | 0.684 | +0.023 | TM |
| OSULeaf | time | 6 | OnlineQuantileTrackerBinarizer | 0.707 | ExtraTrees | 0.680 | +0.027 | TM |
| CricketZ | time | 12 | AdaptiveQuantileBinarizer | 0.707 | ExtraTrees | 0.684 | +0.022 | TM |
| CricketY | time | 12 | OnlineQuantileTrackerBinarizer | 0.701 | ExtraTrees | 0.684 | +0.017 | TM |
| MiddlePhalanxOutlineAgeGroup | time | 3 | OnlineUniversalBinarizer | 0.699 | ExtraTrees | 0.704 | -0.005 | ML |
| ECG5000 | time | 5 | GLADEEncoder | 0.693 | LightGBM | 0.660 | +0.033 | TM |
| Computers | time | 2 | OnlineQuantileTrackerBinarizer | 0.673 | RandomForest | 0.633 | +0.040 | TM |
| tinyml-cs | tabu | 3 | SAQT | 0.672 | LightGBM | 0.821 | -0.148 | ML |
| LargeKitchenAppliances | time | 3 | ResonantGradientBinarizerV2 | 0.666 | LightGBM | 0.641 | +0.024 | TM |
| EthanolLevel | time | 4 | GLADEEncoder | 0.664 | LightGBM | 0.664 | +0.000 | TM |
| ShakeGestureWiimoteZ | time | 10 | OnlineBollingerBinarizer | 0.661 | ExtraTrees | 0.632 | +0.029 | TM |
| Herring | time | 2 | GLADEBooleanizer | 0.654 | XGBoost | 0.666 | -0.012 | ML |
| cic-iot-2023-multiclass | tabu | 8 | ResonantGradientBinarizerV2 | 0.643 | XGBoost | 0.709 | -0.066 | ML |
| DodgerLoopDay | time | 7 | KnownMethodsBinarizer | 0.620 | ExtraTrees | 0.623 | -0.002 | ML |
| EOGVerticalSignal | time | 12 | NTEUniform | 0.618 | ExtraTrees | 0.625 | -0.007 | ML |
| ProximalPhalanxTW | time | 6 | QBEspresso | 0.618 | LightGBM | 0.611 | +0.006 | TM |
| WormsTwoClass | time | 2 | TWINEv3 | 0.613 | XGBoost | 0.545 | +0.068 | TM |
| GestureMidAirD1 | time | 26 | NTEUniform | 0.601 | RandomForest | 0.575 | +0.026 | TM |
| RefrigerationDevices | time | 3 | OnlineGeneralizedBinarizer | 0.600 | RandomForest | 0.598 | +0.002 | TM |
| PLAID | time | 11 | GLADEBooleanizer | 0.599 | RandomForest | 0.528 | +0.072 | TM |
| WordSynonyms | time | 25 | OnlineQuantileTrackerBinarizer | 0.581 | ExtraTrees | 0.650 | -0.069 | ML |
| hikari-2021-multiclass | tabu | 6 | OnlineQuantileSignalBinarizer | 0.576 | LightGBM | 0.615 | -0.039 | ML |
| ShapeletSim | time | 2 | SSL | 0.569 | LightGBM | 0.483 | +0.086 | TM |
| Earthquakes | time | 2 | MWAB | 0.568 | LightGBM | 0.557 | +0.011 | TM |
| cic-iomt-2024-multiclass | tabu | 19 | OnlineGeneralizedBinarizer | 0.553 | XGBoost | 0.557 | -0.003 | ML |
| Worms | time | 5 | SDQB | 0.552 | ExtraTrees | 0.505 | +0.047 | TM |
| InlineSkate | time | 7 | MovingWindowBinarizerV2 | 0.546 | ExtraTrees | 0.569 | -0.023 | ML |
| FiftyWords | time | 50 | TWINELite | 0.545 | ExtraTrees | 0.599 | -0.053 | ML |
| ScreenType | time | 3 | MovingWindowBinarizerV2 | 0.533 | RandomForest | 0.507 | +0.026 | TM |
| GestureMidAirD2 | time | 26 | SAQT | 0.527 | ExtraTrees | 0.536 | -0.009 | ML |
| SemgHandMovementCh2 | time | 6 | PulseResonanceBinarizer | 0.521 | ExtraTrees | 0.518 | +0.003 | TM |
| Haptics | time | 5 | PulseResonanceBinarizer | 0.520 | XGBoost | 0.494 | +0.026 | TM |
| DistalPhalanxTW | time | 6 | DriftRobustBinarizer | 0.508 | XGBoost | 0.550 | -0.042 | ML |
| AllGestureWiimoteY | time | 10 | TWINEv3 | 0.502 | ExtraTrees | 0.610 | -0.107 | ML |
| safe-advent-2025 | tabu | 4 | PulseResonanceBinarizer | 0.495 | ExtraTrees | 0.355 | +0.140 | TM |
| PickupGestureWiimoteZ | time | 10 | KnownMethodsBinarizer | 0.489 | ExtraTrees | 0.538 | -0.049 | ML |
| AllGestureWiimoteZ | time | 10 | NTEUniform | 0.471 | ExtraTrees | 0.479 | -0.008 | ML |
| deep-semantic | tabu | 10 | KBinsThermometer | 0.464 | XGBoost | 0.465 | -0.000 | ML |
| MiddlePhalanxTW | time | 6 | OnlineBollingerBinarizer | 0.422 | RandomForest | 0.383 | +0.039 | TM |
| GestureMidAirD3 | time | 26 | SketchGK | 0.422 | ExtraTrees | 0.346 | +0.076 | TM |
| AllGestureWiimoteX | time | 10 | ACFB | 0.405 | ExtraTrees | 0.460 | -0.055 | ML |
| hynetsys | tabu | 3 | GLADEBooleanizer | 0.400 | XGBoost | 0.445 | -0.045 | ML |
| cic-malmem-2022-multiclass | tabu | 16 | GLADEEncoder | 0.362 | XGBoost | 0.466 | -0.104 | ML |
| PigAirwayPressure | time | 52 | ACFB | 0.103 | ExtraTrees | 0.097 | +0.006 | TM |
| Phoneme | time | 39 | DualDynamicsBinarizer | 0.099 | XGBoost | 0.064 | +0.035 | TM |
| PigArtPressure | time | 52 | SingleSpeedP2 | 0.075 | ExtraTrees | 0.143 | -0.069 | ML |
| PigCVP | time | 52 | SingleSpeedP2 | 0.017 | ExtraTrees | 0.125 | -0.109 | ML |
