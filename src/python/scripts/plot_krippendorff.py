import json
import os
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'output_final', 'grade', 'krippendorff_topic_all_7.5.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'graphs', 'grade', 'threshold_7_5')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_dimension(dimension_name, data, global_alpha=None):
    # Sort topics for consistent plotting
    # Sort by alpha value descending
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    topics = [item[0] for item in sorted_items]
    alphas = [item[1] for item in sorted_items]

    plt.figure(figsize=(15, 8))  # Wide figure for many topics

    # Create bar chart
    bars = plt.bar(topics, alphas, color='skyblue', label='Topic Alpha')

    # Add global average line if available
    if global_alpha is not None:
        plt.axhline(y=global_alpha, color='r', linestyle='--', linewidth=2, label=f'Global Alpha ({global_alpha:.4f})')

    plt.xlabel('Topic ID')
    plt.ylabel('Krippendorff Alpha')
    plt.title(f'Krippendorff Alpha by Topic: {dimension_name}')

    # Rotate x labels to prevent overlap
    plt.xticks(rotation=90, fontsize=8)

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'{dimension_name}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved graph for {dimension_name} to {output_path}")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: File not found at {JSON_PATH}")
        return

    print(f"Loading data from {JSON_PATH}...")
    data = load_data(JSON_PATH)

    # Dimensions to look for
    dimensions = [
        "correctness_topical",
        "coherence_logical",
        "coherence_stylistic",
        "coverage_broad",
        "coverage_deep",
        "consistency_internal",
        "quality_overall"
    ]

    # Structure to hold data for plotting: { dimension: { topic: alpha } }
    plot_data = {dim: {} for dim in dimensions}
    global_alphas = {}

    # Parse JSON
    for key, value in data.items():
        # Check if it's a global alpha
        if key.startswith("alpha_") and key != "alpha_total":
            # Extract dimension name from key e.g. alpha_correctness_topical -> correctness_topical
            dim_name = key.replace("alpha_", "")
            if dim_name in dimensions:
                global_alphas[dim_name] = value

        # Check if it's a topic (value is a dict and has one of the dimensions)
        elif isinstance(value, dict) and any(d in value for d in dimensions):
            topic_id = key
            for dim in dimensions:
                if dim in value and isinstance(value[dim], dict) and "alpha" in value[dim]:
                    # Some values might be None if calculation failed, handle safely
                    alpha_val = value[dim]["alpha"]
                    if alpha_val is not None:
                        plot_data[dim][topic_id] = alpha_val

    # Generate plots
    print(f"Generating graphs in {OUTPUT_DIR}...")
    count = 0
    for dim in dimensions:
        if plot_data[dim]:
            plot_dimension(dim, plot_data[dim], global_alphas.get(dim))
            count += 1
        else:
            print(f"No data found for dimension: {dim}")

    print(f"Done. Generated {count} graphs.")

if __name__ == "__main__":
    main()

