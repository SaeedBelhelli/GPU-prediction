**GPU Performance Modeling & Recommendation**

Small project where I play with GPU specs and try to model performance for different use-cases (AI, gaming, balanced). It uses a curated dataset of recent NVIDIA RTX 30/40-series and AMD RX 6000/7000-series cards, some basic feature engineering, a random forest regressor, and a 3D Plotly visualization.

---

### Requirements

* Python 3.8+
* `pip`

Python packages:

* `pandas`
* `plotly`
* `scikit-learn`

You can install them with:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install pandas plotly scikit-learn
```

### Usage

Run the main script:

```bash
python gpu_ml_tool.py
```

It will:

1. Build a DataFrame from the GPU specs.
2. Engineer a few features (VRAM in GB, clock in GHz, a simple compute proxy, perf per watt).
3. Train a `RandomForestRegressor` to predict a rough `Perf_Index` from the specs.
4. Print:

   * train/test sizes
   * R², MAE, RMSE
   * feature importances
   * average scores per tier
5. Ask how you want to rank GPUs:

   * `ai`
   * `gaming`
   * `balanced`
   * `ml` (ML-predicted performance)
6. Show a 3D Plotly scatter of the GPUs in hardware space, colored by the chosen score.

### What it does

* Uses a small hand-made dataset of real GPUs (RTX 30/40, RX 6000/7000).
* Adds engineered features:

  * `VRAM_GB`
  * `Clock_GHz`
  * `Compute_Proxy` (cores × GHz)
  * `Perf_Per_Watt`
* Trains a random forest to predict a `Perf_Index` (0–100 style performance score) from those features.
* Computes heuristic scores:

  * `AI_Score` → weights VRAM, bandwidth, compute, efficiency
  * `Gaming_Score` → weights compute, clock, bandwidth, VRAM
  * `Balanced_Score` → average of AI + Gaming
* Adds `ML_Predicted_Perf` from the trained model.
* CLI helper to list top GPUs for a given use-case and optional:

  * manufacturer filter (NVIDIA / AMD)
  * max TDP limit
* 3D Plotly scatter:

  * x: memory bandwidth
  * y: VRAM
  * z: compute proxy
  * color: any score (AI / gaming / balanced / ML)

### Customization

* Add or remove GPUs by editing `GPU_DATA`.
* Change the `Perf_Index` values if you want to tune the relative performance scale.
* Adjust the heuristic score weights in `add_heuristic_scores` if you want different priorities for AI vs gaming.
* Swap the model in `train_perf_model` (e.g. try linear regression, gradient boosting, etc.).

### License

MIT.
