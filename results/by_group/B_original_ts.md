# Original 12 time-series

**12 datasets** · TM ≥ ML on **10/12** (83%) · mean macro-F1 **TM 0.9420** vs **ML 0.9303** (Δ +0.0117)

| Dataset | C | TM best booleanizer | TM F1 | ML best | ML F1 | Δ | winner |
|---|--:|---|--:|---|--:|--:|---|
| [Plane](../datasets/Plane.md) | 7 | AdaptiveGaussian | 1.000 | LightGBM | 0.984 | +0.016 | TM |
| [SonyAIBORobotSurface1](../datasets/SonyAIBORobotSurface1.md) | 2 | GLADEBooleanizer | 1.000 | ExtraTrees | 0.995 | +0.005 | TM |
| [Trace](../datasets/Trace.md) | 4 | DecisionTreeBinarizer | 1.000 | ExtraTrees | 0.967 | +0.033 | TM |
| [Wafer](../datasets/Wafer.md) | 2 | AdaptiveQuantileBinarizer | 0.999 | XGBoost | 0.998 | +0.001 | TM |
| [TwoLeadECG](../datasets/TwoLeadECG.md) | 2 | GLADEEncoder | 0.997 | LightGBM | 0.997 | +0.000 | TM |
| [ItalyPowerDemand](../datasets/ItalyPowerDemand.md) | 2 | OnlineATRBinarizer | 0.973 | RandomForest | 0.970 | +0.003 | TM |
| [PowerCons](../datasets/PowerCons.md) | 2 | AdaptiveGaussian | 0.972 | XGBoost | 0.963 | +0.009 | TM |
| [MoteStrain](../datasets/MoteStrain.md) | 2 | DriftRobustBinarizer | 0.968 | ExtraTrees | 0.971 | -0.003 | ML |
| [GunPoint](../datasets/GunPoint.md) | 2 | AdaptiveQuantileBinarizer | 0.967 | RandomForest | 0.967 | +0.000 | TM |
| [ECG200](../datasets/ECG200.md) | 2 | ACFB | 0.923 | ExtraTrees | 0.870 | +0.053 | TM |
| [FordA](../datasets/FordA.md) | 2 | OnlineQuantileTrackerBinarizer | 0.812 | LightGBM | 0.823 | -0.011 | ML |
| [ECG5000](../datasets/ECG5000.md) | 5 | GLADEEncoder | 0.693 | LightGBM | 0.660 | +0.033 | TM |
