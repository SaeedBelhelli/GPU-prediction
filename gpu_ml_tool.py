from __future__ import annotations

from typing import Literal, Optional, Dict, Tuple

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Basic GPU data + rough performance index (0–100)
GPU_DATA = [
    # 40-series
    {"model": "RTX 4090", "manufacturer": "NVIDIA", "tier": "Enthusiast",
     "VRAM_MB": 24576, "Clock_Speed_MHz": 2235, "Cache_KB": 73728, "Cores": 16384,
     "Memory_Bandwidth_GBps": 1008, "TDP_W": 450, "Perf_Index": 100},
    {"model": "RTX 4080", "manufacturer": "NVIDIA", "tier": "High-End",
     "VRAM_MB": 16384, "Clock_Speed_MHz": 2205, "Cache_KB": 65536, "Cores": 9728,
     "Memory_Bandwidth_GBps": 716, "TDP_W": 320, "Perf_Index": 88},
    {"model": "RTX 4070 Ti", "manufacturer": "NVIDIA", "tier": "High-End",
     "VRAM_MB": 12288, "Clock_Speed_MHz": 2310, "Cache_KB": 49152, "Cores": 7680,
     "Memory_Bandwidth_GBps": 504, "TDP_W": 285, "Perf_Index": 78},
    {"model": "RTX 4070", "manufacturer": "NVIDIA", "tier": "High-End",
     "VRAM_MB": 12288, "Clock_Speed_MHz": 1920, "Cache_KB": 36864, "Cores": 5888,
     "Memory_Bandwidth_GBps": 504, "TDP_W": 200, "Perf_Index": 72},
    {"model": "RTX 4060 Ti", "manufacturer": "NVIDIA", "tier": "Mid-Range",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 2310, "Cache_KB": 32768, "Cores": 4352,
     "Memory_Bandwidth_GBps": 288, "TDP_W": 160, "Perf_Index": 60},
    {"model": "RTX 4060", "manufacturer": "NVIDIA", "tier": "Mid-Range",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 1830, "Cache_KB": 24576, "Cores": 3072,
     "Memory_Bandwidth_GBps": 272, "TDP_W": 115, "Perf_Index": 50},

    # 30-series
    {"model": "RTX 3090 Ti", "manufacturer": "NVIDIA", "tier": "Enthusiast",
     "VRAM_MB": 24576, "Clock_Speed_MHz": 1860, "Cache_KB": 6144, "Cores": 10752,
     "Memory_Bandwidth_GBps": 1008, "TDP_W": 450, "Perf_Index": 90},
    {"model": "RTX 3090", "manufacturer": "NVIDIA", "tier": "Enthusiast",
     "VRAM_MB": 24576, "Clock_Speed_MHz": 1695, "Cache_KB": 6144, "Cores": 10496,
     "Memory_Bandwidth_GBps": 936, "TDP_W": 350, "Perf_Index": 86},
    {"model": "RTX 3080 Ti", "manufacturer": "NVIDIA", "tier": "High-End",
     "VRAM_MB": 12288, "Clock_Speed_MHz": 1665, "Cache_KB": 6144, "Cores": 10240,
     "Memory_Bandwidth_GBps": 912, "TDP_W": 350, "Perf_Index": 83},
    {"model": "RTX 3080", "manufacturer": "NVIDIA", "tier": "High-End",
     "VRAM_MB": 10240, "Clock_Speed_MHz": 1710, "Cache_KB": 5120, "Cores": 8704,
     "Memory_Bandwidth_GBps": 760, "TDP_W": 320, "Perf_Index": 79},
    {"model": "RTX 3070 Ti", "manufacturer": "NVIDIA", "tier": "High-End",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 1770, "Cache_KB": 4096, "Cores": 6144,
     "Memory_Bandwidth_GBps": 608, "TDP_W": 290, "Perf_Index": 70},
    {"model": "RTX 3070", "manufacturer": "NVIDIA", "tier": "High-End",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 1725, "Cache_KB": 4096, "Cores": 5888,
     "Memory_Bandwidth_GBps": 448, "TDP_W": 220, "Perf_Index": 65},
    {"model": "RTX 3060 Ti", "manufacturer": "NVIDIA", "tier": "Mid-Range",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 1665, "Cache_KB": 4096, "Cores": 4864,
     "Memory_Bandwidth_GBps": 448, "TDP_W": 200, "Perf_Index": 58},
    {"model": "RTX 3060", "manufacturer": "NVIDIA", "tier": "Mid-Range",
     "VRAM_MB": 12288, "Clock_Speed_MHz": 1777, "Cache_KB": 3072, "Cores": 3584,
     "Memory_Bandwidth_GBps": 360, "TDP_W": 170, "Perf_Index": 48},

    # RDNA3 (7000-series)
    {"model": "RX 7900 XTX", "manufacturer": "AMD", "tier": "Enthusiast",
     "VRAM_MB": 24576, "Clock_Speed_MHz": 2300, "Cache_KB": 98304, "Cores": 12288,
     "Memory_Bandwidth_GBps": 960, "TDP_W": 355, "Perf_Index": 92},
    {"model": "RX 7900 XT", "manufacturer": "AMD", "tier": "High-End",
     "VRAM_MB": 20480, "Clock_Speed_MHz": 2000, "Cache_KB": 81920, "Cores": 10752,
     "Memory_Bandwidth_GBps": 800, "TDP_W": 300, "Perf_Index": 89},
    {"model": "RX 7800 XT", "manufacturer": "AMD", "tier": "High-End",
     "VRAM_MB": 16384, "Clock_Speed_MHz": 2124, "Cache_KB": 65536, "Cores": 7680,
     "Memory_Bandwidth_GBps": 624, "TDP_W": 263, "Perf_Index": 80},
    {"model": "RX 7700 XT", "manufacturer": "AMD", "tier": "High-End",
     "VRAM_MB": 12288, "Clock_Speed_MHz": 2171, "Cache_KB": 49152, "Cores": 3840,
     "Memory_Bandwidth_GBps": 432, "TDP_W": 245, "Perf_Index": 72},
    {"model": "RX 7600", "manufacturer": "AMD", "tier": "Mid-Range",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 2250, "Cache_KB": 32768, "Cores": 2048,
     "Memory_Bandwidth_GBps": 288, "TDP_W": 165, "Perf_Index": 55},

    # RDNA2 (6000-series)
    {"model": "RX 6950 XT", "manufacturer": "AMD", "tier": "Enthusiast",
     "VRAM_MB": 16384, "Clock_Speed_MHz": 2310, "Cache_KB": 132096, "Cores": 5120,
     "Memory_Bandwidth_GBps": 576, "TDP_W": 335, "Perf_Index": 84},
    {"model": "RX 6900 XT", "manufacturer": "AMD", "tier": "Enthusiast",
     "VRAM_MB": 16384, "Clock_Speed_MHz": 2250, "Cache_KB": 132096, "Cores": 5120,
     "Memory_Bandwidth_GBps": 512, "TDP_W": 300, "Perf_Index": 81},
    {"model": "RX 6800 XT", "manufacturer": "AMD", "tier": "High-End",
     "VRAM_MB": 16384, "Clock_Speed_MHz": 2250, "Cache_KB": 132096, "Cores": 4608,
     "Memory_Bandwidth_GBps": 512, "TDP_W": 300, "Perf_Index": 78},
    {"model": "RX 6800", "manufacturer": "AMD", "tier": "High-End",
     "VRAM_MB": 16384, "Clock_Speed_MHz": 2105, "Cache_KB": 131072, "Cores": 3840,
     "Memory_Bandwidth_GBps": 512, "TDP_W": 250, "Perf_Index": 74},
    {"model": "RX 6750 XT", "manufacturer": "AMD", "tier": "Mid-Range",
     "VRAM_MB": 12288, "Clock_Speed_MHz": 2600, "Cache_KB": 98304, "Cores": 2560,
     "Memory_Bandwidth_GBps": 432, "TDP_W": 250, "Perf_Index": 68},
    {"model": "RX 6700 XT", "manufacturer": "AMD", "tier": "Mid-Range",
     "VRAM_MB": 12288, "Clock_Speed_MHz": 2581, "Cache_KB": 98304, "Cores": 2560,
     "Memory_Bandwidth_GBps": 384, "TDP_W": 230, "Perf_Index": 64},
    {"model": "RX 6650 XT", "manufacturer": "AMD", "tier": "Mid-Range",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 2635, "Cache_KB": 32768, "Cores": 2048,
     "Memory_Bandwidth_GBps": 280, "TDP_W": 180, "Perf_Index": 56},
    {"model": "RX 6600 XT", "manufacturer": "AMD", "tier": "Mid-Range",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 2589, "Cache_KB": 32768, "Cores": 2048,
     "Memory_Bandwidth_GBps": 256, "TDP_W": 160, "Perf_Index": 52},
    {"model": "RX 6600", "manufacturer": "AMD", "tier": "Mid-Range",
     "VRAM_MB": 8192, "Clock_Speed_MHz": 2491, "Cache_KB": 32768, "Cores": 1792,
     "Memory_Bandwidth_GBps": 224, "TDP_W": 132, "Perf_Index": 45},
]


UseCase = Literal["ai", "gaming", "balanced", "ml"]


def load_gpu_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(GPU_DATA)
    df["VRAM_GB"] = df["VRAM_MB"] / 1024.0
    df["Clock_GHz"] = df["Clock_Speed_MHz"] / 1000.0
    df["Compute_Proxy"] = df["Cores"] * df["Clock_GHz"]
    df["Perf_Per_Watt"] = df["Compute_Proxy"] / df["TDP_W"]
    return df


def _minmax_normalize(series: pd.Series) -> pd.Series:
    min_val = float(series.min())
    max_val = float(series.max())
    if max_val == min_val:
        return pd.Series(1.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def add_heuristic_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["norm_vram"] = _minmax_normalize(df["VRAM_GB"])
    df["norm_compute"] = _minmax_normalize(df["Compute_Proxy"])
    df["norm_bandwidth"] = _minmax_normalize(df["Memory_Bandwidth_GBps"])
    df["norm_efficiency"] = _minmax_normalize(df["Perf_Per_Watt"])
    df["norm_clock"] = _minmax_normalize(df["Clock_GHz"])

    df["AI_Score"] = (
        0.40 * df["norm_vram"]
        + 0.30 * df["norm_compute"]
        + 0.20 * df["norm_bandwidth"]
        + 0.10 * df["norm_efficiency"]
    ) * 100.0

    df["Gaming_Score"] = (
        0.40 * df["norm_compute"]
        + 0.30 * df["norm_clock"]
        + 0.20 * df["norm_bandwidth"]
        + 0.10 * df["norm_vram"]
    ) * 100.0

    df["Balanced_Score"] = 0.5 * (df["AI_Score"] + df["Gaming_Score"])

    for col in ("AI_Score", "Gaming_Score", "Balanced_Score"):
        df[f"{col}_Rank"] = df[col].rank(ascending=False).astype(int)

    return df


def train_perf_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, dict]:
    feature_cols = [
        "VRAM_GB",
        "Cores",
        "Clock_GHz",
        "Memory_Bandwidth_GBps",
        "Perf_Per_Watt",
    ]

    X = df[feature_cols]
    y = df["Perf_Index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importances": dict(zip(feature_cols, model.feature_importances_)),
    }

    return model, metrics


USE_CASE_TO_COL: Dict[UseCase, str] = {
    "ai": "AI_Score",
    "gaming": "Gaming_Score",
    "balanced": "Balanced_Score",
    "ml": "ML_Predicted_Perf",
}


def recommend_gpus(
    df: pd.DataFrame,
    use_case: UseCase = "ai",
    top_n: int = 5,
    manufacturer: Optional[str] = None,
    max_tdp: Optional[int] = None,
) -> pd.DataFrame:
    if use_case not in USE_CASE_TO_COL:
        valid = ", ".join(USE_CASE_TO_COL.keys())
        raise ValueError(f"Unknown use_case {use_case!r}. Choose from: {valid}")

    score_col = USE_CASE_TO_COL[use_case]
    result = df.copy()

    if manufacturer:
        result = result[result["manufacturer"].str.lower() == manufacturer.lower()]

    if max_tdp is not None:
        result = result[result["TDP_W"] <= max_tdp]

    result = result.sort_values(score_col, ascending=False)
    return result.head(top_n)


def create_gpu_figure(df: pd.DataFrame, color_by: str = "ML_Predicted_Perf"):
    if color_by not in df.columns:
        raise ValueError(f"{color_by!r} is not a column in the DataFrame.")

    fig = px.scatter_3d(
        df,
        x="Memory_Bandwidth_GBps",
        y="VRAM_GB",
        z="Compute_Proxy",
        color=color_by,
        size="TDP_W",
        size_max=22,
        opacity=0.85,
        hover_name="model",
        hover_data={
            "manufacturer": True,
            "tier": True,
            "VRAM_GB": ":.1f",
            "Clock_GHz": ":.2f",
            "Cores": True,
            "Memory_Bandwidth_GBps": ":.0f",
            "TDP_W": ":.0f",
            "Perf_Index": ":.1f",
            "AI_Score": ":.1f",
            "Gaming_Score": ":.1f",
            "Balanced_Score": ":.1f",
            "ML_Predicted_Perf": ":.1f",
        },
        labels={
            "Memory_Bandwidth_GBps": "Memory Bandwidth (GB/s)",
            "VRAM_GB": "VRAM (GB)",
            "Compute_Proxy": "Relative Compute (cores × GHz)",
            color_by: color_by.replace("_", " "),
        },
        title="GPU hardware vs heuristic / ML scores",
        symbol="manufacturer",
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Memory Bandwidth (GB/s)",
            yaxis_title="VRAM (GB)",
            zaxis_title="Relative Compute (cores × GHz)",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend_title_text="Manufacturer",
    )

    return fig


def _interactive_cli(df: pd.DataFrame) -> None:
    print("\n=== GPU Recommendation Helper ===")
    use_case = input("Rank by [ai / gaming / balanced / ml] (default: ai): ").strip().lower()
    if use_case == "":
        use_case = "ai"

    manufacturer = input("Preferred manufacturer [NVIDIA / AMD / leave empty]: ").strip()
    if manufacturer == "":
        manufacturer = None

    max_tdp_str = input("Maximum power (TDP) in watts [optional]: ").strip()
    max_tdp = int(max_tdp_str) if max_tdp_str else None

    try:
        recs = recommend_gpus(df, use_case=use_case, manufacturer=manufacturer, max_tdp=max_tdp)
    except ValueError as e:
        print(f"Error: {e}")
        return

    score_col = USE_CASE_TO_COL[use_case]
    print(f"\nTop {len(recs)} GPUs ranked by '{score_col}':")
    for row in recs.itertuples(index=False):
        score = getattr(row, score_col)
        print(
            f" - {row.model:15} | {row.manufacturer:7} | "
            f"{row.VRAM_GB:4.0f} GB VRAM | {row.Memory_Bandwidth_GBps:4.0f} GB/s | "
            f"TDP {row.TDP_W:3.0f} W | {score_col} = {score:5.1f}"
        )


def main() -> None:
    df = load_gpu_dataframe()
    df = add_heuristic_scores(df)

    model, metrics = train_perf_model(df)
    df["ML_Predicted_Perf"] = model.predict(
        df[["VRAM_GB", "Cores", "Clock_GHz", "Memory_Bandwidth_GBps", "Perf_Per_Watt"]]
    )

    print("GPU dataset loaded.")
    print(f"Total GPUs: {len(df)}\n")

    print("=== ML model evaluation (predicting Perf_Index from specs) ===")
    print(f"Train size: {metrics['train_size']}, Test size: {metrics['test_size']}")
    print(f"R^2 score:   {metrics['r2']:.3f}")
    print(f"MAE:         {metrics['mae']:.2f} Perf_Index points")
    print(f"RMSE:        {metrics['rmse']:.2f} Perf_Index points")
    print("\nFeature importances:")
    for feat, importance in metrics["feature_importances"].items():
        print(f" - {feat:24} {importance:6.3f}")

    print("\nAverage scores by tier:")
    print(
        df.groupby("tier")[["AI_Score", "Gaming_Score", "Balanced_Score", "ML_Predicted_Perf"]]
        .mean()
        .round(1)
        .sort_values("ML_Predicted_Perf", ascending=False)
    )

    _interactive_cli(df)

    fig = create_gpu_figure(df, color_by="ML_Predicted_Perf")
    fig.show()


if __name__ == "__main__":
    main()
