import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os
import argparse
import scienceplots
import ast

plt.style.use("science")
font = {"family": "normal", "weight": "bold", "size": 14}

matplotlib.rc("font", **font)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot model loading time")
    parser.add_argument("-i", "--input_file", type=str, default="profiles.csv", help="Input file")
    return parser.parse_args()

def plot_loading_time(df):
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Scatter plot between load_duration_ms on y axis and onnx_mparams on x axis 
    plt.figure(figsize=(12, 8))
    

    
    # Use seaborn scatterplot with different markers for each family
    families = df['model_family'].unique()
    markers = ['o', 's', '^', 'v', 'D', '*', 'p', 'h', '<', '>']
    
    # Use scienceplot colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df['model_family'] == family]
        sns.scatterplot(
            data=family_data,
            x='onnx_mparams',
            y=family_data['onnx_load_duration_ms'] / 1000,
            label=family,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=150,
            alpha=0.7,
            edgecolor='black',
            linewidth=1.0
        )
    
    plt.xlabel(r"$\mathbf{Param\ (log\ millions)}$", fontweight='bold', fontsize=22)
    plt.ylabel(r"$\mathbf{Loading\ Time\ (log\ s)}$", fontweight='bold', fontsize=22)
    plt.yscale('log')  # Log scale for y-axis
    plt.yticks([0.1, 1, 10, 100])  # Custom tick values
    plt.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, labelsize=20)
    plt.tick_params(axis='both', which='minor', direction='out', length=3, width=1.0)
    # Remove ticks on top and right axes
    ax = plt.gca()
    ax.tick_params(axis='x', which='both', top=False)
    ax.tick_params(axis='y', which='both', right=False)
    
    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    # Thicken the axis lines and hide top/right spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xscale('log')  # Log scale for x-axis
    plt.title("", fontweight='bold')
    plt.legend(loc='upper left', ncol=2, frameon=True, fancybox=True, 
               facecolor='white', edgecolor='black', framealpha=1.0, prop={'weight': 'bold', 'size': 20})
    plt.grid(True, alpha=0.3, which='both')  # Grid for both major and minor ticks
    plt.tight_layout()
    plt.savefig("plots/loading_time_vs_model_size.png", dpi=300, bbox_inches='tight')
    print("Plot saved as plots/loading_time_vs_model_size.png")

def parse_response_time(df):
    parsed_data = []
    for model_variant, response_latencies_str in df.iterrows():
        # Convert string representation of list to actual list
        latencies_list = ast.literal_eval(response_latencies_str['response_latencies'])
        # Convert to milliseconds and add model identifier
        for latency in latencies_list:
            parsed_data.append({
                'model_variant': model_variant,
                'response_time_ms': latency * 1000  # Convert to milliseconds
            })
    
    return pd.DataFrame(parsed_data)

def plot_response_time():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Read the response latencies file
    df = pd.read_csv("response_latencies_A2.csv", index_col=0)
    
    # Parse the string representations of lists into actual numerical data
    df_parsed = parse_response_time(df)
    
    # Create DataFrame from parsed data
    print(f"Parsed data shape: {df_parsed.shape}")
    print(f"Models: {df_parsed['model_variant'].unique()}")
    
    # Create figure with science plots style
    plt.figure(figsize=(12, 8)) 
    
    # Create side-by-side box plots using seaborn
    sns.boxplot(data=df_parsed, x='model_variant', y='response_time_ms', 
                showfliers=False, whis=float('inf'), palette="pastel")
    
    # Customize the plot
    plt.ylabel(r"$\mathbf{Response\ Time\ (ms)}$", fontweight='bold', fontsize=22)
    plt.xlabel(r"$\mathbf{Model\ Variant}$", fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, labelsize=20)
    plt.tick_params(axis='both', which='minor', direction='out', length=3, width=1.0)
    ax = plt.gca()
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    # Set custom y-axis ticks with 10ms intervals
    y_ticks = [0, 10, 20, 30, 40, 50]
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2', 'ResNet50', 'Resnet101'])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    # Thicken the axis lines and hide top/right spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig("plots/response_time_A2.png", dpi=300, bbox_inches='tight')
    print("Response time box plot saved as plots/response_time_A2.png")
    # plt.show()

def plot_response_time_colocation():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Read the response latencies file
    df = pd.read_csv("response_latencies_A2.csv", index_col=0)
    df_effnet = pd.read_csv("response_latencies_A2_colocation.csv", index_col=0)
    df_resnet = pd.read_csv("response_latencies_A2_colocation_resnet.csv", index_col=0)
    
    # Parse the string representations of lists into actual numerical data
    df_effnet = parse_response_time(df_effnet)
    df_resnet = parse_response_time(df_resnet)
    df = parse_response_time(df)

    # Filter for the models we want and add colocation data
    df_final = df[df['model_variant'].isin(['efficientnet_b0', 'resnet50'])].copy()
    
    # Add colocation data
    df_effnet_colocation = df_effnet[df_effnet['model_variant'] == 'efficientnet_b0'].copy()
    df_effnet_colocation['model_variant'] = 'efficientnet_b0_colocation'
    
    df_resnet_colocation = df_resnet[df_resnet['model_variant'] == 'efficientnet_b0'].copy()
    df_resnet_colocation['model_variant'] = 'resnet50_colocation'
    
    # Combine all data
    df_final = pd.concat([df_final, df_effnet_colocation, df_resnet_colocation], ignore_index=True)
        
    # Create figure with two subplots side by side, sharing y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # First subplot: efficientnet_b0 and resnet50 (standalone)
    df_standalone = df_final[df_final['model_variant'].isin(['efficientnet_b0', 'resnet50'])]
    sns.boxplot(data=df_standalone, x='model_variant', y='response_time_ms', 
                showfliers=False, whis=float('inf'), palette="pastel", ax=ax1)
    
    # Second subplot: colocation models
    df_colocation = df_final[df_final['model_variant'].isin(['efficientnet_b0_colocation', 'resnet50_colocation'])]
    sns.boxplot(data=df_colocation, x='model_variant', y='response_time_ms', 
                showfliers=False, whis=float('inf'), palette="pastel", ax=ax2)
    
    # Customize first subplot
    ax1.set_ylabel(r"$\mathbf{Processing\ Latency\ (ms)}$", fontweight='bold', fontsize=20)
    ax1.set_xlabel(r"$\mathbf{(a)In\ Isolation}$", fontweight='bold', fontsize=20)
    # Set custom x-tick labels for first subplot
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['EfficientNet_B0', 'ResNet50'])
    # Set y-axis to start from 0
    ax1.set_ylim(bottom=0)
    # Set custom y-axis ticks with 10ms intervals
    y_ticks = [0, 10, 20, 30, 40, 50]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    # Customize second subplot
    # ax2.set_ylabel("")  # Remove y-axis label
    ax2.set_xlabel(r"$\mathbf{(b)Co-located\ Models}$", fontweight='bold', fontsize=20)
    # Set custom x-tick labels for second subplot
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['EfficientNet_B0', 'ResNet50'])
    # Set y-axis to start from 0
    ax2.set_ylim(bottom=0)
    # Set custom y-axis ticks with 10ms intervals
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_ticks)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontweight('bold')
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    
    plt.subplots_adjust(wspace=0.0)  # Stick subplots together with no space
    plt.savefig("plots/response_time_A2_colocation.png", dpi=300, bbox_inches='tight')
    print("Response time box plot saved as plots/response_time_A2_colocation.png")
    # plt.show()

def main():
    args = parse_args()
    # df = pd.read_csv(args.input_file)
    # plot_loading_time(df)

    plot_response_time()
    # plot_response_time_colocation()

if __name__ == "__main__":
    main()

