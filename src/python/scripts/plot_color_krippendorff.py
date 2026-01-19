import json
import os
import matplotlib.pyplot as plt
import numpy as np
import random

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'output_final', 'comp_fully', 'krippendorff_topic_all.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'graphs', 'comp_fully', 'color')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}

def get_distinct_colors(n):
    # Use nipy_spectral for high variance
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i) for i in np.linspace(0.05, 0.95, n)]
    # Shuffle to avoid similar colors being next to each other
    seed = 42
    random.seed(seed)
    random.shuffle(colors)
    return colors

def plot_graph(topics_subset, filename, title_suffix, topic_data, dimensions):
    num_topics = len(topics_subset)
    if num_topics == 0:
        return

    print(f"Plotting {num_topics} topics into {filename}...")
    colors = get_distinct_colors(num_topics)

    plt.figure(figsize=(24, 14))
    x = np.arange(len(dimensions))

    for idx, topic in enumerate(topics_subset):
        values = topic_data[topic]
        # Plot line connecting dimensions
        plt.plot(x, values,
                 marker='s',        # Square marker
                 markersize=8,
                 linewidth=2,
                 color=colors[idx],
                 label=topic,
                 alpha=0.8,
                 markeredgecolor='black',
                 markeredgewidth=0.5)

    plt.xticks(x, dimensions, rotation=45, ha='right')
    plt.ylabel('Alpha Value')
    plt.title(f'Krippendorff Alpha {title_suffix}')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Legend omit logic if too many
    if num_topics <= 40:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize='small', title="Topics")
    else:
        print(f"Legend omitted due to high number of topics ({num_topics}).")

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Graph saved to {out_file}")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: File not found at {JSON_PATH}")
        return

    print(f"Loading data from {JSON_PATH}...")
    data = load_data(JSON_PATH)

    # Dimensions on X-axis (Fixed order)
    dimensions = [
        "correctness_topical",
        "coherence_logical",
        "coherence_stylistic",
        "coverage_broad",
        "coverage_deep",
        "consistency_internal",
        "quality_overall"
    ]

    # Parse data: topic -> [values]
    topic_data = {}
    for key, value in data.items():
        # Identify topics: must be dict and have at least one of our dimensions
        # and key is not one of the global alpha summaries
        if isinstance(value, dict) and any(d in value for d in dimensions) and not key.startswith("alpha_"):
            topic_id = key
            values = []
            for dim in dimensions:
                # Extract alpha value
                if dim in value and isinstance(value[dim], dict) and 'alpha' in value[dim]:
                    val = value[dim]['alpha']
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)

            # Keep if not all NaN
            if not all(np.isnan(v) for v in values):
                topic_data[topic_id] = values

    if not topic_data:
        print("No valid topic data found.")
        return

    topics = list(topic_data.keys())
    topics.sort()

    # 1. Plot ALL topics
    plot_graph(topics, 'krippendorff_color_plot_all.png', '(All Topics)', topic_data, dimensions)

    # 2. Plot chunks of 13
    CHUNK_SIZE = 13
    total_topics = len(topics)

    for i in range(0, total_topics, CHUNK_SIZE):
        chunk_topics = topics[i:i + CHUNK_SIZE]
        start_idx = i + 1
        end_idx = i + len(chunk_topics)
        filename = f'krippendorff_color_plot_{start_idx}-{end_idx}.png'
        title = f'(Topics {start_idx}-{end_idx})'
        plot_graph(chunk_topics, filename, title, topic_data, dimensions)


if __name__ == "__main__":
    main()