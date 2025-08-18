import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# import seaborn as sns
import numpy as np
import os
import argparse
import scienceplots

plt.style.use("science")
font = {"family": "normal", "weight": "bold", "size": 12}

matplotlib.rc("font", **font)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot model loading time")
    parser.add_argument("-i", "--input_file", type=str, default="profiles.csv", help="Input file")
    return parser.parse_args()

def plot_loading_time(df):
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Scatter plot between load_duration_ms on y axis and onnx_mparams on x axis 
    plt.figure(figsize=(12, 6))
    
    # Group by model family for different colors
    families = df['model_family'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    markers = ['o', 's', '^', 'v', 'D', '*', 'p', 'h', '<', '>']
    
    for i, family in enumerate(families):
        family_data = df[df['model_family'] == family]
        # family_data['onnx_load_duration_s'] = family_data['onnx_load_duration_ms'] / 1000
        plt.scatter(
            family_data['onnx_mparams'], 
            family_data['onnx_load_duration_ms'], 
            label=family, 
            color=colors[i],
            alpha=0.7,
            s=100,
            marker=markers[i]
        )
    
    plt.xlabel("Model Parameters (millions)")
    plt.ylabel("Load Duration (ms)")
    # plt.yscale('log')  # Base-2 logarithmic scale
    # plt.xscale('log')  # Base-2 logarithmic scale
    plt.title("Model Loading Time vs Model Size")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/loading_time_vs_model_size.png", dpi=300, bbox_inches='tight')
    print("Plot saved as plots/loading_time_vs_model_size.png")


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file, index_col=0)

    plot_loading_time(df)

if __name__ == "__main__":
    main()

