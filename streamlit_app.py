import streamlit as st
import pandas as pd

from gpu_ml_tool import (
    load_gpu_dataframe,
    add_heuristic_scores,
    train_perf_model,
    recommend_gpus,
    create_gpu_figure,
    USE_CASE_TO_COL,
)


def prepare_data():
    df = load_gpu_dataframe()
    df = add_heuristic_scores(df)
    model, metrics = train_perf_model(df)
    df["ML_Predicted_Perf"] = model.predict(
        df[["VRAM_GB", "Cores", "Clock_GHz", "Memory_Bandwidth_GBps", "Perf_Per_Watt"]]
    )
    return df, metrics


def main():
    st.set_page_config(page_title="GPU Performance Tool", layout="wide")
    st.title("GPU Performance Modeling & Recommendation")

    df, metrics = prepare_data()

    # ----- metrics / summary -----
    st.subheader("Model evaluation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Train size", metrics["train_size"])
    col2.metric("Test size", metrics["test_size"])
    col3.metric("R² score", f"{metrics['r2']:.3f}")

    st.write(
        f"MAE: **{metrics['mae']:.2f}** · RMSE: **{metrics['rmse']:.2f}** (Perf_Index points)"
    )

    st.write("**Feature importances:**")
    fi = pd.DataFrame(
        {"feature": list(metrics["feature_importances"].keys()),
         "importance": list(metrics["feature_importances"].values())}
    ).sort_values("importance", ascending=False)
    st.bar_chart(fi.set_index("feature"))

    st.markdown("---")

    # ----- controls -----
    st.subheader("Recommendation controls")

    use_case = st.selectbox(
        "Rank by",
        options=["ai", "gaming", "balanced", "ml"],
        format_func=lambda x: {
            "ai": "AI score",
            "gaming": "Gaming score",
            "balanced": "Balanced score",
            "ml": "ML‑predicted performance",
        }[x],
    )

    manufacturer = st.selectbox(
        "Manufacturer",
        options=["Any", "NVIDIA", "AMD"],
    )
    manufacturer_arg = None if manufacturer == "Any" else manufacturer

    max_tdp = st.slider("Max power (TDP, W)", 100, 500, 450, step=10)
    top_n = st.slider("Number of GPUs to show", 3, 10, 5)

    score_col = USE_CASE_TO_COL[use_case]

    recs = recommend_gpus(
        df,
        use_case=use_case,
        top_n=top_n,
        manufacturer=manufacturer_arg,
        max_tdp=max_tdp,
    )

    st.subheader("Recommended GPUs")
    st.dataframe(
        recs[
            [
                "model",
                "manufacturer",
                "tier",
                "VRAM_GB",
                "Memory_Bandwidth_GBps",
                "TDP_W",
                score_col,
            ]
        ].rename(columns={"VRAM_GB": "VRAM (GB)", "Memory_Bandwidth_GBps": "Mem BW (GB/s)"})
    )

    # ----- 3D plot -----
    st.subheader("3D hardware / performance view")
    fig = create_gpu_figure(df, color_by=score_col)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
