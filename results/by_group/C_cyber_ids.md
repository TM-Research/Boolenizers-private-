# Cyber / IDS multi-class

**20 datasets** · TM ≥ ML on **7/20** (35%) · mean macro-F1 **TM 0.7721** vs **ML 0.7878** (Δ -0.0156)

| Dataset | C | TM best booleanizer | TM F1 | ML best | ML F1 | Δ | winner |
|---|--:|---|--:|---|--:|--:|---|
| [5gcid-multiclass](../datasets/5gcid-multiclass.md) | 4 | ACFB | 1.000 | XGBoost | 1.000 | +0.000 | TM |
| [domain-info-2024-multiclass](../datasets/domain-info-2024-multiclass.md) | 3 | GLADEBooleanizer | 1.000 | XGBoost | 1.000 | +0.000 | TM |
| [smart-digital](../datasets/smart-digital.md) | 10 | KBinsThermometer | 0.995 | RandomForest | 0.998 | -0.003 | ML |
| [sd-iot](../datasets/sd-iot.md) | 10 | KalmanFilterBinarizer | 0.995 | XGBoost | 0.999 | -0.004 | ML |
| [cicmaldroid-2020-multiclass](../datasets/cicmaldroid-2020-multiclass.md) | 5 | OnlineRSIMACDBinarizer | 0.990 | LightGBM | 0.951 | +0.039 | TM |
| [cic-iov-2024-multiclass](../datasets/cic-iov-2024-multiclass.md) | 6 | TWINEv2 | 0.979 | XGBoost | 0.975 | +0.004 | TM |
| [ddos-tnsm](../datasets/ddos-tnsm.md) | 7 | AdaptiveMomentumBinarizer | 0.963 | LightGBM | 0.965 | -0.002 | ML |
| [cybersoceval-hybrid-analysis-family](../datasets/cybersoceval-hybrid-analysis-family.md) | 5 | DecisionTreeBinarizer | 0.948 | ExtraTrees | 0.857 | +0.091 | TM |
| [edge-iiotset-multiclass](../datasets/edge-iiotset-multiclass.md) | 15 | OnlineGeneralizedBinarizer | 0.908 | XGBoost | 0.974 | -0.065 | ML |
| [nids-bench-2026](../datasets/nids-bench-2026.md) | 13 | AQB | 0.869 | XGBoost | 0.867 | +0.001 | TM |
| [anf-iot](../datasets/anf-iot.md) | 3 | GLADEEncoder | 0.824 | XGBoost | 0.843 | -0.019 | ML |
| [ornl-msu](../datasets/ornl-msu.md) | 37 | StandardBinarizerNative | 0.808 | ExtraTrees | 0.896 | -0.088 | ML |
| [tinyml-cs](../datasets/tinyml-cs.md) | 3 | SAQT | 0.672 | LightGBM | 0.821 | -0.148 | ML |
| [cic-iot-2023-multiclass](../datasets/cic-iot-2023-multiclass.md) | 8 | ResonantGradientBinarizerV2 | 0.643 | XGBoost | 0.709 | -0.066 | ML |
| [hikari-2021-multiclass](../datasets/hikari-2021-multiclass.md) | 6 | OnlineQuantileSignalBinarizer | 0.576 | LightGBM | 0.615 | -0.040 | ML |
| [cic-iomt-2024-multiclass](../datasets/cic-iomt-2024-multiclass.md) | 19 | OnlineGeneralizedBinarizer | 0.553 | XGBoost | 0.557 | -0.004 | ML |
| [safe-advent-2025](../datasets/safe-advent-2025.md) | 4 | PulseResonanceBinarizer | 0.495 | ExtraTrees | 0.355 | +0.140 | TM |
| [deep-semantic](../datasets/deep-semantic.md) | 10 | KBinsThermometer | 0.464 | XGBoost | 0.465 | -0.000 | ML |
| [hynetsys](../datasets/hynetsys.md) | 3 | GLADEBooleanizer | 0.400 | XGBoost | 0.445 | -0.045 | ML |
| [cic-malmem-2022-multiclass](../datasets/cic-malmem-2022-multiclass.md) | 16 | GLADEEncoder | 0.362 | XGBoost | 0.466 | -0.104 | ML |
