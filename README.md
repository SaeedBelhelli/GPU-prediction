# GPU Performance Visualization

This project creates an interactive 3D scatter plot of real GPU specifications and simulated gaming performance using pandas, numpy, and plotly.

## Prerequisites

* Python 3.7 or later
* pip

## Installation

1. Clone or download this repository.
2. Install required packages:

   ```bash
   pip install pandas numpy plotly
   ```

## Usage

1. Open the main script (for example, `gpu_visualization.py`).
2. Run the script in a Python interpreter or Jupyter/Colab notebook.
3. The code will build a DataFrame of GPU specs, calculate a performance score, simulate FPS for several games, and display the plot.

## Features

* combines real GPU hardware specs with a custom performance metric
* simulates FPS for Cyberpunk 2077, Fortnite, Red Dead Redemption 2, and Valorant
* interactive 3D scatter showing VRAM, memory bandwidth, core count, and FPS color scale
* dropdown filters for GPU tier and manufacturer
* annotations and custom layout for dark theme

## Customization

* change or extend the `gpu_data` list with other models
* adjust the performance formula under the `Performance_Score` calculation
* modify the games and FPS formulas in the DataFrame

## License

This code is released under MIT License.
