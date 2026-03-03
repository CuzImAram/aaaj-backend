import json
import os
import matplotlib.pyplot as plt
import numpy as np
import random

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'output_final', 'comp_zeroshot_without_ref', 'krippendorff_topic_majority.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'graphs', 'comp_zeroshot_without_ref', 'majority', 'color')

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

    handles_normal = []
    labels_normal = []
    handles_glowing = []
    labels_glowing = []

    for idx, topic in enumerate(topics_subset):
        values = topic_data[topic]

        # Calculate fluctuation
        valid_values = [v for v in values if not np.isnan(v)]
        diff = 0.0
        if valid_values:
            diff = max(valid_values) - min(valid_values)

        # Determine styling based on fluctuation
        lw = 2
        opacity = 0.8
        is_glowing_flag = False

        if diff > 0.6:
            lw = 7
            opacity = 1.0
        elif diff >= 0.4:
            lw = 4
            opacity = 1.0
        elif diff < 0.3:
            is_glowing_flag = True
            lw = 3
            opacity = 1.0
        else:
            # Normal
            lw = 2
            opacity = 0.8

        c = colors[idx]
        if is_glowing_flag:
            c = '#FFD700'  # Shiny Gold

        # If glowing, draw the "glow" behind the main line
        if is_glowing_flag:
            plt.plot(x, values,
                     linewidth=12,
                     color=c,
                     alpha=0.3)

        # Plot main line
        line, = plt.plot(x, values,
                 marker='s',        # Square marker
                 markersize=8,
                 linewidth=lw,
                 color=c,
                 label=topic,
                 alpha=opacity,
                 markeredgecolor='black',
                 markeredgewidth=0.5)

        if is_glowing_flag:
            handles_glowing.append(line)
            labels_glowing.append(topic)
        else:
            handles_normal.append(line)
            labels_normal.append(topic)

    plt.xticks(x, dimensions, rotation=45, ha='right')
    plt.ylabel('Alpha Value')
    plt.title(f'Krippendorff Alpha {title_suffix}')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Legend 1: Normal Topics (Top Right)
    # Only show if not too many, or if user wants all (but usually omit > 40)
    if num_topics <= 40 and handles_normal:
        leg1 = plt.legend(handles_normal, labels_normal,
                          bbox_to_anchor=(1.01, 1),
                          loc='upper left',
                          borderaxespad=0.,
                          fontsize='small',
                          title="Topics")
        plt.gca().add_artist(leg1)
    elif num_topics > 40:
        print(f"Normal legend omitted due to high number of topics ({num_topics}).")

    # Legend 2: Glowing/Stable Topics (Bottom Right)
    if handles_glowing:
        plt.legend(handles_glowing, labels_glowing,
                   bbox_to_anchor=(1.01, 0),
                   loc='lower left',
                   borderaxespad=0.,
                   fontsize='small',
                   title="Stable (Gold)")
    else:
        # Show legend even if empty
        import matplotlib.lines as mlines
        dummy_handle = mlines.Line2D([], [], color='#FFD700', marker='s', markersize=8, label='(None)')
        plt.legend([dummy_handle], ['(None)'],
                   bbox_to_anchor=(1.01, 0),
                   loc='lower left',
                   borderaxespad=0.,
                   fontsize='small',
                   title="Stable (Gold)")

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_file, format='pdf')
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
    plot_graph(topics, 'krippendorff_color_plot_all.pdf', '(All Topics)', topic_data, dimensions)

    # 2. Plot chunks of 13
    CHUNK_SIZE = 13
    total_topics = len(topics)

    for i in range(0, total_topics, CHUNK_SIZE):
        chunk_topics = topics[i:i + CHUNK_SIZE]
        start_idx = i + 1
        end_idx = i + len(chunk_topics)
        filename = f'krippendorff_color_plot_{start_idx}-{end_idx}.pdf'
        title = f'(Topics {start_idx}-{end_idx})'
        plot_graph(chunk_topics, filename, title, topic_data, dimensions)


if __name__ == "__main__":
    main()