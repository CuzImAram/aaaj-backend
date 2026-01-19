import json
import os
from collections import Counter

def main():
    # Define paths relative to this script or workspace root
    # Assuming script is in src/python/ and running from aaaj-backend root or similar
    # Let's try to find the data file dynamically or assume standard structure

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, 'data', 'raw', 'ratings.json')
    output_path = os.path.join(base_dir, 'data', 'raw', 'ratings_majority.json')
    output_path_not_majority = os.path.join(base_dir, 'data', 'raw', 'ratings_not_majority.json')

    print(f"Reading from: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dimensions = [
        "correctness_topical",
        "coherence_logical",
        "coherence_stylistic",
        "consistency_internal",
        "coverage_broad",
        "coverage_deep",
        "quality_overall"
    ]

    filtered_data = []
    not_majority_data = []

    for entry in data:
        dimensions_with_majority = 0

        for dim in dimensions:
            vote_key = f"{dim}_vote"
            if vote_key in entry:
                votes = entry[vote_key]
                # Filter out None or invalid votes if any, though usually strings like "A", "B", "N"
                # The votes are in a list of 5 elements
                valid_votes = [v for v in votes if v]

                if not valid_votes:
                    continue

                counter = Counter(valid_votes)
                # check if any vote has >= 3
                # counter.most_common(1) returns [(vote, count)]
                if counter:
                    most_common = counter.most_common(1)[0]
                    count = most_common[1]
                    if count >= 3:
                        dimensions_with_majority += 1

        # Check if majority of utility dimensions (4 out of 7) have a majority vote
        if dimensions_with_majority >= 4:
            filtered_data.append(entry)
        else:
            not_majority_data.append(entry)

    print(f"Total entries: {len(data)}")
    print(f"Filtered entries (Majority): {len(filtered_data)}")
    print(f"Discarded entries (Not Majority): {len(not_majority_data)}")

    print(f"Writing to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Writing to: {output_path_not_majority}")

    with open(output_path_not_majority, 'w', encoding='utf-8') as f:
        json.dump(not_majority_data, f, indent=4)

if __name__ == "__main__":
    main()
