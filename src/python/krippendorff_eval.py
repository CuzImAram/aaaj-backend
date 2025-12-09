"""
Krippendorff's Alpha evaluation module for measuring agreement between agent and gold ratings.

This module provides functionality to calculate Krippendorff's alpha coefficient
for inter-rater reliability between agent judgements and human gold standard ratings.
"""

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import krippendorff

from data_sender import REPO_ROOT
from ratings_eval import (
    load_ratings,
    compare_by_ids,
    DEFAULT_FIELDS,
    RATINGS_PATH,
    COMPARISON_OUTPUT_DIR
)

# Paths
KRIPPENDORFF_OUTPUT_DIR = REPO_ROOT / "data" / "output" / "krippendorff"

# Default threshold for grade comparisons
DEFAULT_THRESHOLD = 7.5

logger = logging.getLogger(__name__)


def get_output_filename(
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    prefix: str = "krippendorff",
    topic_ids: Optional[List[str]] = None,
    is_all_topics: bool = False,
    is_all_ratings: bool = False
) -> str:
    """Generate output filename based on mode.

    Args:
        output_dir: Directory where krippendorff files are saved.
        prefix: Base prefix for the filename.
        topic_ids: List of topic IDs for topic mode.
        is_all_topics: True if evaluating all topics.
        is_all_ratings: True if evaluating all ratings.

    Returns:
        Filename string like "krippendorff_topic_2024-5957.json" or "krippendorff_global_all.json".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_all_ratings:
        # Global all ratings mode
        return "krippendorff_global_all.json"
    elif is_all_topics:
        # All topics mode
        return "krippendorff_topic_all.json"
    elif topic_ids:
        # Specific topic(s) - join topic IDs with underscore
        topic_suffix = "_".join(sorted(topic_ids))
        return f"krippendorff_topic_{topic_suffix}.json"
    else:
        # Regular mode - use numbered sequence
        i = 0
        while True:
            filename = f"{prefix}_{i}.json"
            if not (output_dir / filename).exists():
                return filename
            i += 1


def ensure_ratings_exist(response_pairs: List[Tuple[str, str]], threshold: float = DEFAULT_THRESHOLD) -> None:
    """Ensure all comparison files exist with the correct threshold, creating missing ones in parallel.

    If a comparison file exists but has a different threshold, it will be deleted
    and recreated with the correct threshold.

    Args:
        response_pairs: List of (response_a_id, response_b_id) tuples.
        threshold: The threshold to use for grade comparisons. Files with a different
                  threshold will be deleted and recreated.
    """
    missing_pairs = []
    for response_a_id, response_b_id in response_pairs:
        filename = f"{response_a_id}_{response_b_id}.json"
        comparison_file = COMPARISON_OUTPUT_DIR / filename

        if comparison_file.exists():
            # Check if the file has the correct threshold
            try:
                with comparison_file.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                existing_threshold = existing_data.get("threshold")

                if existing_threshold != threshold:
                    logger.info(
                        "Comparison %s has threshold %.2f, but %.2f is required. Deleting and recreating.",
                        filename, existing_threshold if existing_threshold else 0, threshold
                    )
                    comparison_file.unlink()
                    missing_pairs.append((response_a_id, response_b_id))
            except Exception as exc:
                logger.warning("Failed to read threshold from %s: %s. Will recreate.", filename, exc)
                try:
                    comparison_file.unlink()
                except Exception:
                    pass
                missing_pairs.append((response_a_id, response_b_id))
        else:
            missing_pairs.append((response_a_id, response_b_id))

    if missing_pairs:
        logger.info("Creating %d missing comparison file(s) in parallel", len(missing_pairs))

        def create_one(pair: Tuple[str, str]) -> None:
            """Create a single comparison file."""
            response_a_id, response_b_id = pair
            try:
                compare_by_ids(
                    response_a_id,
                    response_b_id,
                    compare_with_gold=False,
                    threshold=threshold
                )
                logger.info("Created comparison: %s vs %s", response_a_id, response_b_id)
            except Exception as exc:
                logger.error("Failed to create comparison %s vs %s: %s",
                           response_a_id, response_b_id, exc)

        # Use ThreadPoolExecutor to create comparisons in parallel
        max_workers = min(len(missing_pairs), 10)  # Limit concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(create_one, pair): pair for pair in missing_pairs}

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Exception creating comparison for %s: %s", pair, exc)

        # Verify files were created
        still_missing = []
        for response_a_id, response_b_id in missing_pairs:
            filename = f"{response_a_id}_{response_b_id}.json"
            comparison_file = COMPARISON_OUTPUT_DIR / filename
            if not comparison_file.exists():
                still_missing.append((response_a_id, response_b_id))

        if still_missing:
            logger.warning("Failed to create comparisons for: %s", still_missing)
    else:
        logger.debug("All %d comparison file(s) already exist", len(response_pairs))


def load_comparison(response_a_id: str, response_b_id: str) -> Optional[dict]:
    """Load a comparison file for the given response pair.

    Args:
        response_a_id: ID of response A.
        response_b_id: ID of response B.

    Returns:
        The comparison dict, or None if it cannot be loaded.
    """
    filename = f"{response_a_id}_{response_b_id}.json"
    comparison_file = COMPARISON_OUTPUT_DIR / filename

    try:
        with comparison_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load comparison %s: %s", filename, exc)
        return None


def get_gold_winner(response_a_id: str, response_b_id: str, field: str) -> Optional[str]:
    """Get gold winner for a specific pair and field from ratings.json.

    Args:
        response_a_id: ID of response A.
        response_b_id: ID of response B.
        field: Field name (e.g., "correctness_topical").

    Returns:
        Gold winner ('a', 'b', or 'n'), or None if not found.
    """
    try:
        ratings = load_ratings()

        # Find the rating entry for this pair
        for entry in ratings:
            if (entry.get("response_a") == response_a_id and
                entry.get("response_b") == response_b_id):
                gold_key = f"{field}_gold"
                gold_value = entry.get(gold_key)
                if gold_value:
                    return gold_value.lower()
                else:
                    logger.warning("No gold value for field %s in pair %s vs %s",
                                 field, response_a_id, response_b_id)
                    return None

        logger.warning("No rating entry found for pair %s vs %s", response_a_id, response_b_id)
        return None

    except Exception as exc:
        logger.error("Error loading gold winner: %s", exc)
        return None


def get_all_topics(ratings_path: Path = RATINGS_PATH) -> List[str]:
    """Get all unique topic IDs (query_id) from ratings.json.

    Args:
        ratings_path: Path to ratings.json.

    Returns:
        List of unique topic IDs.
    """
    ratings = load_ratings(ratings_path)
    topics = set()
    for entry in ratings:
        query_id = entry.get("query_id")
        if query_id:
            topics.add(query_id)
    return sorted(list(topics))


def get_pairs_for_topic(topic_id: str, ratings_path: Path = RATINGS_PATH) -> List[Tuple[str, str]]:
    """Get all response pairs for a specific topic.

    Args:
        topic_id: The topic ID (query_id) to filter by.
        ratings_path: Path to ratings.json.

    Returns:
        List of (response_a_id, response_b_id) tuples for the topic.
    """
    ratings = load_ratings(ratings_path)
    pairs = []
    for entry in ratings:
        if entry.get("query_id") == topic_id:
            response_a_id = entry.get("response_a")
            response_b_id = entry.get("response_b")
            if response_a_id and response_b_id:
                pairs.append((response_a_id, response_b_id))
    return pairs


def get_pairs_for_topics(topic_ids: List[str], ratings_path: Path = RATINGS_PATH) -> List[Tuple[str, str]]:
    """Get all response pairs for multiple topics.

    Args:
        topic_ids: List of topic IDs (query_id) to filter by.
        ratings_path: Path to ratings.json.

    Returns:
        List of (response_a_id, response_b_id) tuples for all topics.
    """
    ratings = load_ratings(ratings_path)
    topic_set = set(topic_ids)
    pairs = []
    for entry in ratings:
        if entry.get("query_id") in topic_set:
            response_a_id = entry.get("response_a")
            response_b_id = entry.get("response_b")
            if response_a_id and response_b_id:
                pairs.append((response_a_id, response_b_id))
    return pairs


def calculate_alpha_for_field(
    comparisons: List[dict],
    field: str
) -> Tuple[Optional[float], List[Tuple[str, str]]]:
    """Calculate Krippendorff's alpha for a specific field.

    Args:
        comparisons: List of comparison dicts.
        field: Field name to calculate alpha for.

    Returns:
        Tuple of (alpha value or None, list of (gold_value, agent_value) pairs).
    """
    gold_values = []
    agent_values = []
    value_pairs = []

    for comp in comparisons:
        response_a_id = comp.get("response_a")
        response_b_id = comp.get("response_b")

        fields_data = comp.get("fields", {})
        field_data = fields_data.get(field)

        if field_data is None:
            logger.warning("Field %s not found in comparison", field)
            continue

        # Get agent winner from comparison file
        agent_winner = field_data.get("agent_winner")

        # Get gold winner from ratings.json
        gold_winner = get_gold_winner(response_a_id, response_b_id, field)

        if gold_winner is not None and agent_winner is not None:
            gold_values.append(gold_winner)
            agent_values.append(agent_winner)
            value_pairs.append((gold_winner, agent_winner))

    # Need at least 2 data points
    if len(gold_values) < 2:
        logger.warning("Not enough data for field %s (need at least 2, got %d)",
                      field, len(gold_values))
        return None, value_pairs

    # Map values to numbers: a=0, b=1, n=2
    mapping = {'a': 0, 'b': 1, 'n': 2}

    try:
        gold_numeric = [mapping[v] for v in gold_values]
        agent_numeric = [mapping[v] for v in agent_values]
    except KeyError as exc:
        logger.error("Invalid value in field %s: %s", field, exc)
        return None, value_pairs

    # Create reliability data matrix (2 coders x n items)
    reliability_data = np.array([gold_numeric, agent_numeric])

    # Calculate Krippendorff's alpha
    try:
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement='nominal'
        )
        return alpha, value_pairs
    except Exception as exc:
        logger.error("Error calculating alpha for field %s: %s", field, exc)
        return None, value_pairs


def evaluate_krippendorff_by_topics(
    topic_ids_list: List[str],
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    is_all_topics: bool = False,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha with per-topic breakdown.

    Args:
        topic_ids_list: List of topic IDs to evaluate.
        fields: List of fields to evaluate. If None, uses DEFAULT_FIELDS.
        ratings_path: Path to ratings.json.
        output_dir: Directory where output will be saved.
        is_all_topics: True if evaluating all topics.
        show_comparisons: If True, include comparison arrays in output. If False, omit them.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to the saved krippendorff JSON file.
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all pairs for these topics
    all_pairs_by_topic = {}
    for topic_id in topic_ids_list:
        pairs = get_pairs_for_topic(topic_id, ratings_path)
        if pairs:
            all_pairs_by_topic[topic_id] = pairs

    if not all_pairs_by_topic:
        raise ValueError(f"No pairs found for topics {topic_ids_list}")

    # Flatten all pairs for ensure_ratings_exist
    all_pairs = []
    for pairs in all_pairs_by_topic.values():
        all_pairs.extend(pairs)

    # Ensure all comparison files exist (creates missing ones in parallel)
    ensure_ratings_exist(all_pairs, threshold)

    # Calculate alpha for each topic separately
    results = {}
    topic_details = {}
    overall_field_alphas = {field: [] for field in fields}

    for topic_id, pairs in all_pairs_by_topic.items():
        # Load comparisons for this topic
        comparisons = []
        comparison_ids = []

        for response_a_id, response_b_id in pairs:
            comp = load_comparison(response_a_id, response_b_id)
            if comp is not None:
                comparisons.append(comp)
                comparison_ids.append(f"{response_a_id}_{response_b_id}")

        if not comparisons:
            logger.warning("No valid comparisons for topic %s", topic_id)
            continue

        # Calculate alpha for each field for this topic
        topic_field_details = {}
        topic_alphas = []

        for field in fields:
            alpha, value_pairs = calculate_alpha_for_field(comparisons, field)

            if alpha is not None:
                topic_alphas.append(alpha)
                overall_field_alphas[field].append(alpha)
                field_detail = {"alpha": round(alpha, 4)}
                if show_comparisons:
                    pairs_list = [[gold, agent] for gold, agent in value_pairs]
                    field_detail["comparisons"] = pairs_list
                topic_field_details[field] = field_detail
            else:
                topic_field_details[field] = {
                    "alpha": None,
                    "error": "Insufficient data or calculation error"
                }

        # Calculate topic total alpha
        topic_total = round(np.mean(topic_alphas), 4) if topic_alphas else None

        topic_detail = {
            "alpha_total": topic_total,
            "n_comparisons": len(comparisons)
        }
        if show_comparisons:
            topic_detail["comparisons"] = comparison_ids
        topic_detail.update(topic_field_details)

        topic_details[topic_id] = topic_detail

    # Calculate overall alpha for each field (average across topics)
    for field in fields:
        field_key = f"alpha_{field}"
        if overall_field_alphas[field]:
            results[field_key] = round(np.mean(overall_field_alphas[field]), 4)
        else:
            results[field_key] = None

    # Calculate overall total alpha
    all_field_alphas = [v for vals in overall_field_alphas.values() for v in vals]
    if all_field_alphas:
        results["alpha_total"] = round(np.mean(all_field_alphas), 4)
    else:
        results["alpha_total"] = None

    # Add topic details after overall alphas
    results.update(topic_details)

    # Save to file
    filename = get_output_filename(output_dir, topic_ids=topic_ids_list, is_all_topics=is_all_topics)
    output_path = output_dir / filename

    # Format JSON with compact comparison arrays
    placeholders = {}
    dumpable = dict(results)

    # Replace comparisons arrays with placeholders for each topic
    placeholder_idx = 0
    for topic_id, topic_data in topic_details.items():
        if topic_id in dumpable:
            dumpable[topic_id] = dict(topic_data)
            for field in fields:
                if field in dumpable[topic_id]:
                    comps = dumpable[topic_id][field].get("comparisons")
                    if comps is not None:
                        placeholder = f"__COMPARISONS_PLACEHOLDER_{placeholder_idx}__"
                        placeholders[placeholder] = comps
                        dumpable[topic_id][field]["comparisons"] = placeholder
                        placeholder_idx += 1

    # Dump to JSON string with pretty indent
    json_str = json.dumps(dumpable, ensure_ascii=False, indent=2)

    # Replace placeholders with compact comparison arrays
    for placeholder, comps in placeholders.items():
        placeholder_json = json.dumps(placeholder, ensure_ascii=False)
        lines = []
        for pair in comps:
            encoded_vals = ",".join(json.dumps(v, ensure_ascii=False) for v in pair)
            lines.append(f"[{encoded_vals}]")
        compact = "[\n" + ",\n".join("        " + ln for ln in lines) + "\n      ]"
        json_str = json_str.replace(placeholder_json, compact)

    # Write final string to file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json_str)

    logger.info("Saved Krippendorff's alpha results to %s", output_path)
    return output_path


def evaluate_krippendorff(
    response_pairs: List[Tuple[str, str]],
    fields: Optional[Sequence[str]] = None,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    prefix: str = "krippendorff",
    topic_ids: Optional[List[str]] = None,
    is_all_topics: bool = False,
    is_all_ratings: bool = False,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for given response pairs.

    Args:
        response_pairs: List of (response_a_id, response_b_id) tuples.
        fields: List of fields to evaluate. If None, uses DEFAULT_FIELDS.
        output_dir: Directory where output will be saved.
        prefix: Prefix for output filename (used only for regular numbered mode).
        topic_ids: List of topic IDs for topic mode filename.
        is_all_topics: True if evaluating all topics.
        is_all_ratings: True if evaluating all ratings.
        show_comparisons: If True, include comparison arrays in output. If False, omit them.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to the saved krippendorff JSON file.
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all comparison files exist (creates missing ones in parallel)
    ensure_ratings_exist(response_pairs, threshold)

    # Load all comparisons
    comparisons = []
    comparison_ids = []

    for response_a_id, response_b_id in response_pairs:
        comp = load_comparison(response_a_id, response_b_id)
        if comp is not None:
            comparisons.append(comp)
            comparison_ids.append(f"{response_a_id}_{response_b_id}")
        else:
            logger.warning("Skipping pair %s vs %s (comparison unavailable)",
                          response_a_id, response_b_id)

    if not comparisons:
        raise ValueError("No valid comparisons loaded")

    # Calculate alpha for each field
    results = {}
    alphas = []
    field_data_map = {}

    for field in fields:
        alpha, value_pairs = calculate_alpha_for_field(comparisons, field)

        field_key = f"alpha_{field}"

        if alpha is not None:
            results[field_key] = round(alpha, 4)
            alphas.append(alpha)
            field_detail = {"alpha": round(alpha, 4)}
            if show_comparisons:
                # Convert value pairs to list of lists for JSON serialization
                pairs_list = [[gold, agent] for gold, agent in value_pairs]
                field_detail["comparisons"] = pairs_list
            field_data_map[field] = field_detail
        else:
            results[field_key] = None
            field_data_map[field] = {
                "alpha": None,
                "error": "Insufficient data or calculation error"
            }

    # Calculate total average alpha
    if alphas:
        results["alpha_total"] = round(np.mean(alphas), 4)
    else:
        results["alpha_total"] = None

    # Add field details directly (no "fields" wrapper)
    for field, data in field_data_map.items():
        results[field] = data

    # Add n_comparisons and optionally comparisons at the end
    results["n_comparisons"] = len(comparisons)
    if show_comparisons:
        results["comparisons"] = comparison_ids

    # Save to file
    filename = get_output_filename(output_dir, prefix, topic_ids, is_all_topics, is_all_ratings)
    output_path = output_dir / filename

    # We want the inner comparison pairs to be on a single line, e.g. ["a","a"].
    # json.dump with indent=2 will pretty-print inner lists across multiple lines.
    # To achieve the desired formatting we replace each field's comparisons with a
    # unique placeholder, dump the JSON, then replace the JSON-encoded placeholder
    # with a compact representation of the comparisons array.
    placeholders = {}
    dumpable = dict(results)  # shallow copy

    # For each field that has comparisons, replace with a placeholder string
    for idx, field in enumerate(fields):
        if field in dumpable and isinstance(dumpable[field], dict):
            comps = dumpable[field].get("comparisons")
            if comps is None:
                continue
            placeholder = f"__COMPARISONS_PLACEHOLDER_{idx}__"
            placeholders[placeholder] = comps
            # set the placeholder (will be JSON-encoded as a string)
            dumpable[field] = dict(dumpable[field])
            dumpable[field]["comparisons"] = placeholder

    # Dump to JSON string with pretty indent
    json_str = json.dumps(dumpable, ensure_ascii=False, indent=2)

    # Replace each JSON-encoded placeholder (including JSON quotes) with the
    # compact comparisons array string (each inner pair on a single line)
    for placeholder, comps in placeholders.items():
        # JSON will encode the placeholder as a JSON string; obtain that
        placeholder_json = json.dumps(placeholder, ensure_ascii=False)

        # Build compact comparisons string: each pair as ["a","b"] on one line
        lines = []
        for pair in comps:
            # make sure to JSON-encode each value (to handle special chars)
            encoded_vals = ",".join(json.dumps(v, ensure_ascii=False) for v in pair)
            lines.append(f"[{encoded_vals}]")

        compact = "[\n" + ",\n".join("        " + ln for ln in lines) + "\n      ]"

        # Replace the encoded placeholder with the compact string
        json_str = json_str.replace(placeholder_json, compact)

    # Write final string to file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json_str)

    logger.info("Saved Krippendorff's alpha results to %s", output_path)
    return output_path


def evaluate_by_ids(
    response_pairs: List[Tuple[str, str]],
    fields: Optional[Sequence[str]] = None,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for specific response ID pairs.

    Args:
        response_pairs: List of (response_a_id, response_b_id) tuples.
        fields: List of fields to evaluate.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    return evaluate_krippendorff(response_pairs, fields, output_dir, show_comparisons=show_comparisons, threshold=threshold)


def evaluate_first_n(
    count: int,
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for the first N rating pairs.

    Args:
        count: Number of pairs to evaluate.
        fields: List of fields to evaluate.
        ratings_path: Path to ratings.json.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    ratings = load_ratings(ratings_path)

    response_pairs = []
    for entry in ratings[:count]:
        response_a_id = entry.get("response_a")
        response_b_id = entry.get("response_b")

        if response_a_id and response_b_id:
            response_pairs.append((response_a_id, response_b_id))

    return evaluate_krippendorff(response_pairs, fields, output_dir, show_comparisons=show_comparisons, threshold=threshold)


def evaluate_random_n(
    count: int,
    random_seed: Optional[int] = None,
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for N random rating pairs.

    Args:
        count: Number of pairs to evaluate.
        random_seed: Optional seed for reproducibility.
        fields: List of fields to evaluate.
        ratings_path: Path to ratings.json.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    ratings = load_ratings(ratings_path)

    rng = random.Random(random_seed)
    selected_entries = rng.sample(ratings, min(count, len(ratings)))

    response_pairs = []
    for entry in selected_entries:
        response_a_id = entry.get("response_a")
        response_b_id = entry.get("response_b")

        if response_a_id and response_b_id:
            response_pairs.append((response_a_id, response_b_id))

    return evaluate_krippendorff(response_pairs, fields, output_dir, show_comparisons=show_comparisons, threshold=threshold)


def evaluate_all(
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for all rating pairs.

    Args:
        fields: List of fields to evaluate.
        ratings_path: Path to ratings.json.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    ratings = load_ratings(ratings_path)

    response_pairs = []
    for entry in ratings:
        response_a_id = entry.get("response_a")
        response_b_id = entry.get("response_b")

        if response_a_id and response_b_id:
            response_pairs.append((response_a_id, response_b_id))

    return evaluate_krippendorff(
        response_pairs,
        fields,
        output_dir,
        is_all_ratings=True,
        show_comparisons=show_comparisons,
        threshold=threshold
    )


def evaluate_topic(
    topic_id: str,
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for all pairs in a specific topic.

    Args:
        topic_id: The topic ID (query_id) to evaluate.
        fields: List of fields to evaluate.
        ratings_path: Path to ratings.json.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    logger.info("Evaluating topic %s", topic_id)
    return evaluate_krippendorff_by_topics(
        [topic_id],
        fields,
        ratings_path,
        output_dir,
        is_all_topics=False,
        show_comparisons=show_comparisons,
        threshold=threshold
    )


def evaluate_topics(
    topic_ids: List[str],
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for all pairs in multiple topics.

    Args:
        topic_ids: List of topic IDs (query_id) to evaluate.
        fields: List of fields to evaluate.
        ratings_path: Path to ratings.json.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    logger.info("Evaluating %d topics", len(topic_ids))
    return evaluate_krippendorff_by_topics(
        topic_ids,
        fields,
        ratings_path,
        output_dir,
        is_all_topics=False,
        show_comparisons=show_comparisons,
        threshold=threshold
    )


def evaluate_random_topic(
    count: int = 1,
    random_seed: Optional[int] = None,
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for random topic(s).

    Args:
        count: Number of random topics to select. If 1, evaluates a single topic.
               If > 1, evaluates multiple topics together.
        random_seed: Optional seed for reproducibility.
        fields: List of fields to evaluate.
        ratings_path: Path to ratings.json.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    all_topics = get_all_topics(ratings_path)

    if not all_topics:
        raise ValueError("No topics found in ratings.json")

    rng = random.Random(random_seed)

    # Select random topic(s)
    num_to_select = min(count, len(all_topics))
    selected_topics = rng.sample(all_topics, num_to_select)

    if count == 1:
        logger.info("Randomly selected topic: %s", selected_topics[0])
        return evaluate_topic(selected_topics[0], fields, ratings_path, output_dir, show_comparisons, threshold)
    else:
        logger.info("Randomly selected %d topics: %s", len(selected_topics), selected_topics)
        return evaluate_topics(selected_topics, fields, ratings_path, output_dir, show_comparisons, threshold)


def evaluate_all_topics(
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD
) -> Path:
    """Evaluate Krippendorff's alpha for all topics combined.

    Args:
        fields: List of fields to evaluate.
        ratings_path: Path to ratings.json.
        output_dir: Output directory.
        show_comparisons: If True, include comparison arrays in output.
        threshold: The threshold to use for grade comparisons (default: 1.0).

    Returns:
        Path to saved results.
    """
    all_topics = get_all_topics(ratings_path)

    if not all_topics:
        raise ValueError("No topics found in ratings.json")

    logger.info("Evaluating all %d topics", len(all_topics))
    return evaluate_krippendorff_by_topics(
        all_topics,
        fields,
        ratings_path,
        output_dir,
        is_all_topics=True,
        show_comparisons=show_comparisons,
        threshold=threshold
    )


def main(
    *,
    count: Optional[int] = None,
    randomize: bool = False,
    random_seed: Optional[int] = None,
    all_ratings: bool = False,
    response_pairs: Optional[List[Tuple[str, str]]] = None,
    # Topic mode parameters
    topic_id: Optional[str] = None,
    topic_ids: Optional[List[str]] = None,
    random_topic: bool = False,
    all_topics: bool = False,
    fields: Optional[Sequence[str]] = None,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = KRIPPENDORFF_OUTPUT_DIR,
    show_comparisons: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    log_level: str = "INFO"
) -> None:
    """Entry point for Krippendorff's alpha evaluation.

    Choose one of the following modes:

    Regular modes:
    - response_pairs: evaluate specific pairs [(a1,b1), (a2,b2), ...]
    - all_ratings=True: evaluate all pairs
    - count + randomize=True: evaluate N random pairs
    - count: evaluate first N pairs

    Topic modes (creates krippendorff_topic_N.json files):
    - topic_id: evaluate all pairs for a specific topic
    - topic_ids: evaluate all pairs for multiple topics
    - random_topic=True: evaluate all pairs for a random topic
    - random_topic=True + count=N: evaluate all pairs for N random topics
    - all_topics=True: evaluate all pairs for all topics

    Args:
        count: Number of pairs/topics to evaluate. Works with randomize and random_topic.
        randomize: If True, evaluate random pairs instead of first N.
        random_seed: Optional seed for reproducible randomness.
        all_ratings: If True, evaluate all rating pairs.
        response_pairs: List of (response_a_id, response_b_id) tuples.
        topic_id: Evaluate all pairs for a specific topic (query_id).
        topic_ids: Evaluate all pairs for multiple topics.
        random_topic: If True, evaluate random topic(s). Use count to specify how many.
        all_topics: If True, evaluate all topics.
        fields: List of fields to evaluate. If None, uses DEFAULT_FIELDS.
        ratings_path: Path to ratings.json.
        output_dir: Directory for output files.
        show_comparisons: If True, include comparison arrays in JSON output. If False, omit them.
        threshold: The threshold to use for grade comparisons (default: 1.0).
                  Files with a different threshold will be deleted and recreated.
        log_level: Logging level (string such as "INFO" or "DEBUG").
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s"
    )

    # Topic modes take priority
    if topic_id:
        output_file = evaluate_topic(topic_id, fields, ratings_path, output_dir, show_comparisons, threshold)
    elif topic_ids:
        output_file = evaluate_topics(topic_ids, fields, ratings_path, output_dir, show_comparisons, threshold)
    elif random_topic:
        topic_count = count if count is not None else 1
        output_file = evaluate_random_topic(topic_count, random_seed, fields, ratings_path, output_dir, show_comparisons, threshold)
    elif all_topics:
        output_file = evaluate_all_topics(fields, ratings_path, output_dir, show_comparisons, threshold)
    # Regular modes
    elif response_pairs:
        output_file = evaluate_by_ids(response_pairs, fields, output_dir, show_comparisons, threshold)
    elif all_ratings:
        output_file = evaluate_all(fields, ratings_path, output_dir, show_comparisons, threshold)
    elif randomize:
        if count is None:
            count = 1
        output_file = evaluate_random_n(count, random_seed, fields, ratings_path, output_dir, show_comparisons, threshold)
    elif count is not None:
        output_file = evaluate_first_n(count, fields, ratings_path, output_dir, show_comparisons, threshold)
    else:
        # Default: evaluate first pair
        output_file = evaluate_first_n(1, fields, ratings_path, output_dir, show_comparisons, threshold)

    logger.info("Krippendorff's alpha evaluation completed: %s", output_file.name)


if __name__ == "__main__":

    # Multiple topics with comparisons (default)
    # main(topic_ids=["2024-105741", "2024-5957"], show_comparisons=True)

    # Multiple topics WITHOUT comparisons (cleaner output)
    main(topic_ids=["2024-42497", "2024-44544"], threshold= 7.5, fields=["correctness_topical"], show_comparisons=True)

    # Random topic: creates krippendorff_topic_<random-id>.json
    # main(random_topic=True)

    # 10 random topics: creates krippendorff_topic_<id1>_<id2>_..._<id10>.json
    # main(count=10, random_topic=True)

    # All topics: creates krippendorff_topic_all.json
    # main(all_topics=True)

    # All ratings (global): creates krippendorff_global_all.json
    # main(all_ratings=True)

    # Regular mode with specific pairs: creates krippendorff_0.json (numbered)
    # main(response_pairs=[("04d71b5f-a8b0-3ab3-8725-43510f6e21f8", "90f27401-7376-3eea-846c-15d6092292e2"),
    #                      ("158a0f7e-f45b-3bfa-a93a-4733662c2216", "90f27401-7376-3eea-846c-15d6092292e2")])
    # main(count=2, randomize=True)
