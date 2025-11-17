import json
import logging
import os
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_RESPONSES_PATH = REPO_ROOT / "data" / "raw" / "responses.json"
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "agent_judgement"
WEBHOOK_ENV_VAR = "WEBHOOK_N8N_START_AGENTS"
DEFAULT_RANDOM_COUNT = 1
REQUEST_TIMEOUT_SECONDS = 60

PostFn = Callable[[dict], List[dict]]

logger = logging.getLogger(__name__)

#--------------------------------------Helpers---------------------------------------------------------

def _load_env_file(path: Path) -> None:
    """Load environment variables from a simple dotenv-style file.

    The file format expected is one KEY=VALUE pair per line. Lines starting with
    '#' or empty lines are ignored. Variables already present in ``os.environ``
    will not be overwritten.

    Args:
        path: Path to the .env file. If the file does not exist the function
            returns silently.

    Returns:
        None

    Side-effects:
        Sets environment variables in ``os.environ`` for any keys found in the
        file that are not already present.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Only set if not already in environment
                if key and key not in os.environ:
                    os.environ[key] = value


# Load .env.production if WEBHOOK_N8N_START_AGENTS is not already set
if WEBHOOK_ENV_VAR not in os.environ:
    env_file = REPO_ROOT / "src" / ".env.production"
    _load_env_file(env_file)


class ResponseNotFoundError(KeyError):
    """Raised when one or more requested response IDs cannot be found.

    This subclass of ``KeyError`` is thrown by ``select_by_ids`` when the
    caller requests IDs that are not present in the input responses list.
    """


def load_responses(path: Path = RAW_RESPONSES_PATH) -> List[dict]:
    """Load all responses from disk and return them as a list of dicts.

    Args:
        path: Path to the JSON file containing a list of response objects.

    Returns:
        A list of dictionary objects loaded from the JSON file.

    Raises:
        ValueError: if the file does not contain a JSON list.
        FileNotFoundError / JSONDecodeError: propagated from ``json.load`` if
            the file cannot be read or is invalid JSON.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("responses.json is expected to contain a list")
    return data


def _index_by_id(responses: Sequence[dict]) -> dict:
    """Create a mapping from response ID to response entry.

    Args:
        responses: Sequence of response dictionaries. Each entry is expected to
            contain a ``"response"`` key with a unique identifier.

    Returns:
        A dict mapping response_id -> response_dict. Entries without a
        "response" key are skipped.
    """
    index = {}
    for entry in responses:
        response_id = entry.get("response")
        if not response_id:
            continue
        index[response_id] = entry
    return index


def _resolve_response_ids(response_ids: Union[str, Sequence[str]]) -> List[str]:
    """Normalize a single response id or a sequence of ids to a list.

    Args:
        response_ids: Either a single string id or a sequence of string ids.

    Returns:
        A list of response id strings.
    """
    if isinstance(response_ids, str):
        return [response_ids]
    return list(response_ids)


def select_first(responses: Sequence[dict], count: int) -> List[dict]:
    """Select the first ``count`` entries from the provided responses.

    Args:
        responses: Sequence of response dicts.
        count: Number of entries to return. Must be > 0.

    Returns:
        A list containing at most ``count`` response dicts, in original order.

    Raises:
        ValueError: if ``count`` is not greater than zero.
    """
    if count <= 0:
        raise ValueError("count must be greater than zero")
    return list(responses[:count])


def select_random(responses: Sequence[dict], count: int, seed: Optional[int] = None) -> List[dict]:
    """Select ``count`` random unique entries from ``responses``.

    The selection is deterministic when a ``seed`` is provided.

    Args:
        responses: Sequence of response dicts.
        count: Number of entries to sample. Must be > 0 and <= len(responses).
        seed: Optional random seed for deterministic sampling.

    Returns:
        A list of ``count`` randomly sampled response dicts.

    Raises:
        ValueError: if ``count`` <= 0 or greater than the number of available
            responses.
    """
    if count <= 0:
        raise ValueError("count must be greater than zero")
    if count > len(responses):
        raise ValueError("count cannot exceed the number of available responses")
    rng = random.Random(seed)
    return rng.sample(list(responses), count)


def select_by_ids(responses: Sequence[dict], response_ids: Union[str, Sequence[str]]) -> List[dict]:
    """Select responses matching the given response id(s).

    Args:
        responses: Sequence of response dicts to search through.
        response_ids: A single id or a sequence of ids to select.

    Returns:
        A list of response dicts corresponding to the requested ids, in the
        same order as ``response_ids``.

    Raises:
        ResponseNotFoundError: if one or more requested ids are missing.
    """
    requested = _resolve_response_ids(response_ids)
    missing: List[str] = []
    index = _index_by_id(responses)
    selected: List[dict] = []
    for response_id in requested:
        try:
            selected.append(index[response_id])
        except KeyError:
            missing.append(response_id)
    if missing:
        raise ResponseNotFoundError(f"Unknown response IDs: {', '.join(missing)}")
    return selected


def _get_webhook_url() -> str:
    """Return the configured webhook URL from the environment.

    Looks up the environment variable named by ``WEBHOOK_ENV_VAR`` and returns
    its value. If the variable is not set a ``RuntimeError`` is raised which
    signals that the caller must configure the webhook location.

    Returns:
        The webhook URL string.

    Raises:
        RuntimeError: if the environment variable is not set.
    """
    webhook = os.getenv(WEBHOOK_ENV_VAR)
    if not webhook:
        raise RuntimeError(f"Environment variable {WEBHOOK_ENV_VAR} is not set")
    return webhook


def _ensure_output_dir(path: Path) -> None:
    """Ensure the output directory exists.

    Creates the directory (and parents) if necessary.

    Args:
        path: Directory path to ensure exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def _post_payload(payload: dict) -> List[dict]:
    """POST a JSON payload to the configured webhook and return parsed result.

    This function performs a HTTP POST with the given payload encoded as JSON
    to the URL obtained from ``_get_webhook_url``. It expects the webhook to
    return either a JSON list of judgement objects, or a JSON object that wraps
    a list under a key such as "data" or "output". In the latter case the
    inner list is returned.

    Args:
        payload: The JSON-serialisable payload to send to the webhook.

    Returns:
        A list of judgement dicts parsed from the webhook response.

    Raises:
        RuntimeError: for HTTP or connection errors.
        ValueError: if the response is not valid JSON or not in the expected
            structure (list or dict wrapping a list).
    """
    webhook_url = _get_webhook_url()
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Webhook HTTP error: {exc.status} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Webhook connection error: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Webhook response is not valid JSON. Received: {body[:200]}") from exc

    # Handle n8n wrapping response in an object (e.g., {"data": [...]})
    if isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ['data', 'output', 'result', 'response']:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # If no wrapper found, treat the dict as a single judgement
        logger.warning("Webhook returned dict, wrapping in list")
        return [parsed]

    if not isinstance(parsed, list):
        raise ValueError(f"Webhook response must be a list or dict, got {type(parsed).__name__}")

    return parsed


def _save_outputs(judgements: Iterable[dict], output_dir: Path) -> List[Path]:
    """Write judgement objects to files in ``output_dir`` and return paths.

    Each judgement dict must contain a "response" key (response_id). The webhook
    returns all fields for a response in a single object. Each judgement is saved
    as a single JSON file directly under output_dir: output/agent_judgement/{response_id}.json

    Args:
        judgements: Iterable of judgement dicts as returned from the webhook.
        output_dir: Directory where judgement files will be written.

    Returns:
        A list of Path objects representing the files written.
    """
    saved_paths: List[Path] = []
    for judgement in judgements:
        response_id = judgement.get("response")

        if not response_id:
            logger.warning("Skipping webhook entry without response id: %s", judgement)
            continue

        # Create filename: response_id.json directly in output_dir
        target = output_dir / f"{response_id}.json"

        # Save the entire judgement object as JSON
        with target.open("w", encoding="utf-8") as handle:
            json.dump(judgement, handle, ensure_ascii=False, indent=2)

        saved_paths.append(target)
        logger.info("Saved %s", target.relative_to(output_dir))

    return saved_paths


def _dispatch_entries(
    entries: Sequence[dict],
    *,
    output_dir: Path,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Send each entry to the webhook (or a provided post function) and save results.

    This function sends all entries to the webhook in parallel using a thread pool.
    Each webhook response (which may itself be a list of judgements) is then 
    persisted to ``output_dir`` using ``_save_outputs``.

    Args:
        entries: Sequence of response payloads to dispatch (each a dict).
        output_dir: Directory where webhook outputs will be saved.
        post_fn: Optional callable taking a payload dict and returning a list
            of judgement dicts. If not provided, ``_post_payload`` is used.
        dry_run: If True, skip POSTing to the webhook (useful for debugging).

    Returns:
        A list of Path objects for all files written.
    """
    _ensure_output_dir(output_dir)
    saved: List[Path] = []
    sender = post_fn or _post_payload
    
    if dry_run:
        for entry in entries:
            response_id = entry.get("response", "<unknown>")
            logger.info("Dry-run enabled, skipping webhook call for %s", response_id)
        return saved
    
    # Send all requests in parallel
    def send_one(entry: dict) -> List[dict]:
        """Send a single entry and return judgements."""
        response_id = entry.get("response", "<unknown>")
        logger.info("Dispatching response %s", response_id)
        try:
            return sender(entry)
        except Exception as exc:
            logger.error("Failed to dispatch response %s: %s", response_id, exc)
            return []
    
    # Use ThreadPoolExecutor to send requests in parallel
    max_workers = min(len(entries), 10)  # Limit concurrent requests to 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {executor.submit(send_one, entry): entry for entry in entries}
        
        # Collect results as they complete
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            response_id = entry.get("response", "<unknown>")
            try:
                judgements = future.result()
                saved.extend(_save_outputs(judgements, output_dir))
                logger.info("Completed response %s", response_id)
            except Exception as exc:
                logger.error("Exception processing response %s: %s", response_id, exc)
    
    return saved

#--------------------------------------Senders---------------------------------------------------------

def send_first_n(
    count: int,
    *,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Send the first N responses from the top of the file.

    Args:
        count: Number of entries to send.
        responses_path: Path to the source responses.json file.
        output_dir: Where to place webhook response files.
        post_fn: Optional post function for testing.
        dry_run: If True, do not actually POST to webhook.

    Returns:
        List of filesystem Paths to saved judgement files.
    """
    responses = load_responses(responses_path)
    entries = select_first(responses, count)
    return _dispatch_entries(entries, output_dir=output_dir, post_fn=post_fn, dry_run=dry_run)


def send_random_n(
    count: int,
    *,
    random_seed: Optional[int] = None,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Send N random responses.

    Args:
        count: Number of random entries to send.
        random_seed: Optional seed for deterministic randomness.
        responses_path: Path to responses.json.
        output_dir: Output directory for judgement files.
        post_fn: Optional post function for testing.
        dry_run: If True, skip actual POST requests.

    Returns:
        List of Paths to saved judgement files.
    """
    responses = load_responses(responses_path)
    entries = select_random(responses, count, seed=random_seed)
    return _dispatch_entries(entries, output_dir=output_dir, post_fn=post_fn, dry_run=dry_run)


def send_by_id(
    response_id: str,
    *,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Send a single response by ID.

    Args:
        response_id: The response identifier to send.
        responses_path: Path to the source responses.json file.
        output_dir: Directory to save webhook judgement files.
        post_fn: Optional post function for testing.
        dry_run: If True, skip network calls.

    Returns:
        List with the Path to the saved judgement file(s).

    Raises:
        ResponseNotFoundError: if the requested id is not present in the
            responses file.
    """
    responses = load_responses(responses_path)
    entries = select_by_ids(responses, response_id)
    return _dispatch_entries(entries, output_dir=output_dir, post_fn=post_fn, dry_run=dry_run)


def send_by_ids(
    response_ids: Sequence[str],
    *,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Send multiple responses by their IDs.

    Args:
        response_ids: Sequence of response identifiers to send.
        responses_path: Path to the source responses.json file.
        output_dir: Directory to save webhook judgement files.
        post_fn: Optional post function for testing.
        dry_run: If True, skip network calls.

    Returns:
        List of Paths to the saved judgement files.

    Raises:
        ResponseNotFoundError: if any requested ids are not present.
    """
    responses = load_responses(responses_path)
    entries = select_by_ids(responses, response_ids)
    return _dispatch_entries(entries, output_dir=output_dir, post_fn=post_fn, dry_run=dry_run)


def main(
    *,
    count: Optional[int] = None,
    randomize: bool = False,
    random_seed: Optional[int] = None,
    response_id: Optional[str] = None,
    response_ids: Optional[Sequence[str]] = None,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> None:
    """Entry point callable without argparse.

    Choose one of the following modes:
    - response_id: send a single response by ID
    - response_ids: send multiple responses by IDs
    - count + randomize=True: send N random responses
    - count: send first N responses

    This function is designed to be called directly from other code (or from
    tests). It does not parse command line arguments; instead keyword
    arguments control its behaviour.

    Args:
        count: Number of responses to send in index-based modes.
        randomize: If True, send random entries instead of the first N.
        random_seed: Optional seed for reproducible randomness.
        response_id: If provided, send only this one response.
        response_ids: If provided, send these specific ids.
        responses_path: Path to the source responses.json file.
        output_dir: Directory where judgement files will be saved.
        dry_run: If True, do not perform network calls.
        log_level: Logging level (string such as "INFO" or "DEBUG").
    """

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if response_id:
        saved = send_by_id(
            response_id,
            responses_path=responses_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )
    elif response_ids:
        saved = send_by_ids(
            response_ids,
            responses_path=responses_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )
    elif randomize:
        if count is None:
            count = DEFAULT_RANDOM_COUNT
        saved = send_random_n(
            count,
            random_seed=random_seed,
            responses_path=responses_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )
    elif count is not None:
        saved = send_first_n(
            count,
            responses_path=responses_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )
    else:
        # Default: send first one
        saved = send_first_n(
            DEFAULT_RANDOM_COUNT,
            responses_path=responses_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )

    logger.info("Saved %d webhook response files", len(saved))


if __name__ == "__main__":
    # Example usage:
    # Send specific IDs:
    main(response_ids=["3c5e25b6-2d4d-3d84-b01c-0264c3a5ba50", "02693406-15df-33d5-b424-219ac8ab2054"])

    # Or send first 5:
    # main(count=5)

    # Or send 5 random:
    # main(count=5, randomize=True)

    # Or send single ID:
    # main(response_id="ae54d7e0-62df-3e53-9bea-3e107a6e5801")
