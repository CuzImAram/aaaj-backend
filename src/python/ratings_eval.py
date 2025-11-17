"""
Rating evaluation module for comparing agent judgements with gold ratings.

This module provides functionality to compare agent-generated judgements against
human gold ratings, computing agreement metrics based on grade comparisons.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Sequence

from data_sender import send_by_id, send_by_ids, REPO_ROOT

# Paths
RATINGS_PATH = REPO_ROOT / "data" / "raw" / "ratings.json"
AGENT_JUDGEMENT_DIR = REPO_ROOT / "data" / "output" / "agent_judgement"
COMPARISON_OUTPUT_DIR = REPO_ROOT / "data" / "output" / "compared_ratings_agent"

# Default fields to compare (based on gold fields in ratings.json)
DEFAULT_FIELDS = [
    "correctness_topical",
    "coherence_logical",
    "coherence_stylistic",
    "coverage_broad",
    "coverage_deep",
    "consistency_internal",
    "quality_overall"
]

logger = logging.getLogger(__name__)


class JudgementNotFoundError(Exception):
    """Raised when a required judgement file cannot be found or created."""


def load_ratings(path: Path = RATINGS_PATH) -> List[dict]:
    """Load all ratings from disk and return them as a list of dicts.

    Args:
        path: Path to the JSON file containing ratings data.

    Returns:
        A list of rating dictionaries.

    Raises:
        ValueError: if the file does not contain a JSON list.
        FileNotFoundError / JSONDecodeError: if file cannot be read.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("ratings.json is expected to contain a list")
    return data


def get_agent_judgement(response_id: str, field: str) -> Optional[dict]:
    """Get agent judgement for a specific response and field.

    Looks in the agent_judgement folder for the flat judgement file
    `{response_id}.json`. If not found, triggers the data_sender to fetch it
    from the webhook. Returns a dict shaped like
    {"output": <field_dict>, "response": response_id, "field": field} for
    uniform downstream consumption.

    Args:
        response_id: The response identifier.
        field: The field name (e.g., "correctness_topical").

    Returns:
        The field-specific judgement dict wrapped with metadata, or None.

    Raises:
        JudgementNotFoundError: if judgement cannot be retrieved.
    """
    judgement_file = AGENT_JUDGEMENT_DIR / f"{response_id}.json"

    def _load_field_from_file(path: Path) -> Optional[dict]:
        try:
            with path.open("r", encoding="utf-8") as f:
                full = json.load(f)
            field_data = full.get(field)
            if field_data is None:
                logger.warning("Field %s not present in %s", field, path.name)
                return None
            return {"output": field_data, "response": response_id, "field": field}
        except Exception as exc:
            logger.error("Failed reading %s: %s", path, exc)
            return None

    # Try existing file first
    if judgement_file.exists():
        logger.debug("Found existing judgement: %s", judgement_file)
        found = _load_field_from_file(judgement_file)
        if found is not None:
            return found

    # Fetch via webhook and retry
    logger.info("Judgement not found or missing field for %s, fetching from webhook", response_id)
    try:
        saved_files = send_by_id(response_id, output_dir=AGENT_JUDGEMENT_DIR)
        for saved_file in saved_files:
            # Expect flat file named {response_id}.json
            if saved_file.name == f"{response_id}.json":
                return _load_field_from_file(saved_file)
        logger.warning("Webhook did not create %s.json for %s", response_id, response_id)
        return None
    except Exception as exc:
        raise JudgementNotFoundError(
            f"Failed to retrieve judgement for {response_id}: {exc}"
        ) from exc


def ensure_judgements_exist(response_ids: List[str]) -> None:
    """Ensure all judgement files exist, fetching missing ones in batch.

    Args:
        response_ids: List of response IDs to check and fetch if needed.
    """
    missing_ids = []
    for response_id in response_ids:
        judgement_file = AGENT_JUDGEMENT_DIR / f"{response_id}.json"
        if not judgement_file.exists():
            missing_ids.append(response_id)

    if missing_ids:
        logger.info("Fetching %d missing judgement(s): %s", len(missing_ids), missing_ids)
        send_by_ids(missing_ids, output_dir=AGENT_JUDGEMENT_DIR)

        # Verify all files were created
        still_missing = []
        for response_id in missing_ids:
            judgement_file = AGENT_JUDGEMENT_DIR / f"{response_id}.json"
            if not judgement_file.exists():
                still_missing.append(response_id)

        if still_missing:
            logger.warning("Failed to fetch judgements for: %s", still_missing)
    else:
        logger.debug("All %d judgement(s) already exist", len(response_ids))


def compare_grades(grade_a: float, grade_b: float, threshold: float) -> str:
    """Compare two grades and determine winner based on threshold.

    Args:
        grade_a: Grade for response A.
        grade_b: Grade for response B.
        threshold: The range within which grades are considered equal.

    Returns:
        "a" if A wins, "b" if B wins, "n" if within threshold (neutral/equal).
    """
    diff = abs(grade_a - grade_b)

    if diff <= threshold:
        return "n"
    elif grade_a > grade_b:
        return "a"
    else:
        return "b"


def compare_pair(
    response_a_id: str,
    response_b_id: str,
    rating_entry: dict,
    fields: Optional[Sequence[str]] = None,
    threshold: float = 2.0,
) -> dict:
    """Compare judgements for a pair of responses across specified fields.

    Args:
        response_a_id: ID of response A.
        response_b_id: ID of response B.
        rating_entry: The rating entry from ratings.json containing gold labels.
        fields: List of fields to compare. If None, uses DEFAULT_FIELDS.
        threshold: Grade difference threshold for equality.

    Returns:
        A dict containing comparison results for all fields.
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    results = {
        "response_a": response_a_id,
        "response_b": response_b_id,
        "threshold": threshold,
        "fields": {}
    }

    for field in fields:
        gold_key = f"{field}_gold"
        gold_value = rating_entry.get(gold_key)

        if gold_value is None:
            logger.warning("No gold value found for field %s", field)
            continue

        # Get agent judgements (will fetch from webhook if not found locally)
        try:
            judgement_a = get_agent_judgement(response_a_id, field)
            judgement_b = get_agent_judgement(response_b_id, field)

            # If either judgement is None after fetching attempt, skip this field
            if judgement_a is None or judgement_b is None:
                missing_responses = []
                if judgement_a is None:
                    missing_responses.append(f"response_a ({response_a_id})")
                if judgement_b is None:
                    missing_responses.append(f"response_b ({response_b_id})")

                logger.warning("Missing judgement for field %s: %s", field, ", ".join(missing_responses))
                results["fields"][field] = {
                    "agent_winner": None,
                    "gold_winner": gold_value.lower(),
                    "match": False,
                    "error": f"Missing judgement for {', '.join(missing_responses)}"
                }
                continue

            # Extract grades
            grade_a = judgement_a.get("output", {}).get("grade")
            grade_b = judgement_b.get("output", {}).get("grade")

            if grade_a is None or grade_b is None:
                missing_grades = []
                if grade_a is None:
                    missing_grades.append(f"response_a ({response_a_id})")
                if grade_b is None:
                    missing_grades.append(f"response_b ({response_b_id})")

                logger.warning("Missing grade in judgement for field %s: %s", field, ", ".join(missing_grades))
                results["fields"][field] = {
                    "agent_winner": None,
                    "gold_winner": gold_value.lower(),
                    "match": False,
                    "error": f"Missing grade for {', '.join(missing_grades)}"
                }
                continue

            # Compare grades
            agent_winner = compare_grades(grade_a, grade_b, threshold)
            gold_winner = gold_value.lower()

            results["fields"][field] = {
                "agent_winner": agent_winner,
                "gold_winner": gold_winner,
                "match": agent_winner == gold_winner,
                "grade_a": grade_a,
                "grade_b": grade_b
            }

        except JudgementNotFoundError as exc:
            logger.error("Error getting judgement for field %s: %s", field, exc)
            results["fields"][field] = {
                "agent_winner": None,
                "gold_winner": gold_value.lower(),
                "match": False,
                "error": str(exc)
            }

    # Calculate overall accuracy
    matches = sum(1 for f in results["fields"].values() if f.get("match", False))
    total = len([f for f in results["fields"].values() if f.get("agent_winner") is not None])

    return results


def save_comparison(results: dict, output_dir: Path = COMPARISON_OUTPUT_DIR) -> Path:
    """Save comparison results to a JSON file.

    Args:
        results: Comparison results dict.
        output_dir: Directory where comparison files will be saved.

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    response_a = results["response_a"]
    response_b = results["response_b"]
    filename = f"{response_a}_{response_b}.json"
    target = output_dir / filename

    with target.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Saved comparison to %s", target.relative_to(output_dir.parent))
    return target


def compare_by_ids(
    response_a_id: str,
    response_b_id: str,
    *,
    fields: Optional[Sequence[str]] = None,
    threshold: float = 2.0,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = COMPARISON_OUTPUT_DIR,
) -> Path:
    """Compare two specific responses by their IDs.

    Args:
        response_a_id: ID of response A.
        response_b_id: ID of response B.
        fields: List of fields to compare. If None, uses DEFAULT_FIELDS.
        threshold: Grade difference threshold for equality.
        ratings_path: Path to ratings.json.
        output_dir: Directory for output files.

    Returns:
        Path to the saved comparison file.

    Raises:
        ValueError: if the pair is not found in ratings.json.
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    # Ensure both judgements exist before comparing
    ensure_judgements_exist([response_a_id, response_b_id])

    ratings = load_ratings(ratings_path)

    # Find the rating entry for this pair
    rating_entry = None
    for entry in ratings:
        if (entry.get("response_a") == response_a_id and
            entry.get("response_b") == response_b_id):
            rating_entry = entry
            break

    if rating_entry is None:
        raise ValueError(
            f"No rating entry found for pair {response_a_id} vs {response_b_id}"
        )

    results = compare_pair(response_a_id, response_b_id, rating_entry, fields, threshold)
    return save_comparison(results, output_dir)


def compare_first_n(
    count: int,
    *,
    fields: Optional[Sequence[str]] = None,
    threshold: float = 2.0,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = COMPARISON_OUTPUT_DIR,
) -> List[Path]:
    """Compare the first N rating pairs.

    Args:
        count: Number of pairs to compare.
        fields: List of fields to compare. If None, uses DEFAULT_FIELDS.
        threshold: Grade difference threshold for equality.
        ratings_path: Path to ratings.json.
        output_dir: Directory for output files.

    Returns:
        List of Paths to saved comparison files.
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    ratings = load_ratings(ratings_path)

    # Collect all response IDs we'll need
    all_response_ids = []
    for entry in ratings[:count]:
        response_a_id = entry.get("response_a")
        response_b_id = entry.get("response_b")
        if response_a_id:
            all_response_ids.append(response_a_id)
        if response_b_id:
            all_response_ids.append(response_b_id)

    # Batch fetch all missing judgements
    ensure_judgements_exist(all_response_ids)

    saved_files = []

    for i, entry in enumerate(ratings[:count]):
        response_a_id = entry.get("response_a")
        response_b_id = entry.get("response_b")

        if not response_a_id or not response_b_id:
            logger.warning("Skipping entry %d: missing response IDs", i)
            continue

        logger.info("Comparing pair %d/%d: %s vs %s", i+1, count, response_a_id, response_b_id)

        results = compare_pair(response_a_id, response_b_id, entry, fields, threshold)
        saved_file = save_comparison(results, output_dir)
        saved_files.append(saved_file)

    return saved_files


def compare_random_n(
    count: int,
    *,
    random_seed: Optional[int] = None,
    fields: Optional[Sequence[str]] = None,
    threshold: float = 2.0,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = COMPARISON_OUTPUT_DIR,
) -> List[Path]:
    """Compare N random rating pairs.

    Args:
        count: Number of pairs to compare.
        random_seed: Optional seed for reproducible randomness.
        fields: List of fields to compare. If None, uses DEFAULT_FIELDS.
        threshold: Grade difference threshold for equality.
        ratings_path: Path to ratings.json.
        output_dir: Directory for output files.

    Returns:
        List of Paths to saved comparison files.
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    ratings = load_ratings(ratings_path)

    rng = random.Random(random_seed)
    selected_entries = rng.sample(ratings, min(count, len(ratings)))

    # Collect all response IDs we'll need
    all_response_ids = []
    for entry in selected_entries:
        response_a_id = entry.get("response_a")
        response_b_id = entry.get("response_b")
        if response_a_id:
            all_response_ids.append(response_a_id)
        if response_b_id:
            all_response_ids.append(response_b_id)

    # Batch fetch all missing judgements
    ensure_judgements_exist(all_response_ids)

    saved_files = []

    for i, entry in enumerate(selected_entries):
        response_a_id = entry.get("response_a")
        response_b_id = entry.get("response_b")

        if not response_a_id or not response_b_id:
            logger.warning("Skipping entry: missing response IDs")
            continue

        logger.info("Comparing pair %d/%d: %s vs %s", i+1, len(selected_entries),
                   response_a_id, response_b_id)

        results = compare_pair(response_a_id, response_b_id, entry, fields, threshold)
        saved_file = save_comparison(results, output_dir)
        saved_files.append(saved_file)

    return saved_files


def main(
    *,
    count: Optional[int] = None,
    randomize: bool = False,
    random_seed: Optional[int] = None,
    response_a_id: Optional[str] = None,
    response_b_id: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
    threshold: float = 2.0,
    ratings_path: Path = RATINGS_PATH,
    output_dir: Path = COMPARISON_OUTPUT_DIR,
    log_level: str = "INFO",
) -> None:
    """Entry point for rating evaluation.

    Choose one of the following modes:
    - response_a_id + response_b_id: compare a specific pair
    - count + randomize=True: compare N random pairs
    - count: compare first N pairs

    Args:
        count: Number of pairs to compare.
        randomize: If True, compare random pairs instead of first N.
        random_seed: Optional seed for reproducible randomness.
        response_a_id: ID of response A for specific pair comparison.
        response_b_id: ID of response B for specific pair comparison.
        fields: List of fields to compare. If None, uses DEFAULT_FIELDS.
        threshold: Grade difference threshold for equality.
        ratings_path: Path to ratings.json.
        output_dir: Directory for output files.
        log_level: Logging level (string such as "INFO" or "DEBUG").
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s"
    )

    if response_a_id and response_b_id:
        saved = [compare_by_ids(
            response_a_id,
            response_b_id,
            fields=fields,
            threshold=threshold,
            ratings_path=ratings_path,
            output_dir=output_dir,
        )]
    elif randomize:
        if count is None:
            count = 1
        saved = compare_random_n(
            count,
            random_seed=random_seed,
            fields=fields,
            threshold=threshold,
            ratings_path=ratings_path,
            output_dir=output_dir,
        )
    elif count is not None:
        saved = compare_first_n(
            count,
            fields=fields,
            threshold=threshold,
            ratings_path=ratings_path,
            output_dir=output_dir,
        )
    else:
        # Default: compare first pair
        saved = compare_first_n(
            1,
            fields=fields,
            threshold=threshold,
            ratings_path=ratings_path,
            output_dir=output_dir,
        )

    logger.info("Saved %d comparison files", len(saved))


if __name__ == "__main__":
    # Example usage:

    # Compare specific pair by IDs:
    #main(
    #    response_a_id="3c5e25b6-2d4d-3d84-b01c-0264c3a5ba50",
    #    response_b_id="02693406-15df-33d5-b424-219ac8ab2054",
    #    fields=["correctness_topical", "coherence_logical"],
    #    threshold=2.0
    #)

    # Other examples:
    # Compare first 5 pairs:
    main(count=5, threshold=2.0, fields=["correctness_topical"], randomize=True)

    # Compare 10 random pairs:
    #main(count=2, randomize=True)
