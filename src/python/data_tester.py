import json
import logging
import os
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_RESPONSES_PATH = REPO_ROOT / "data" / "raw" / "responses.json"
OUTPUT_DIR = REPO_ROOT / "data" / "output"
WEBHOOK_ENV_VAR = "WEBHOOK_N8N_START_AGENTS"
DEFAULT_RANDOM_COUNT = 1
REQUEST_TIMEOUT_SECONDS = 60

PostFn = Callable[[dict], List[dict]]

logger = logging.getLogger(__name__)


def _load_env_file(path: Path) -> None:
    """Load environment variables from a .env file."""
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
    """Raised when a requested response ID cannot be located."""


def load_responses(path: Path = RAW_RESPONSES_PATH) -> List[dict]:
    """Load all responses from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("responses.json is expected to contain a list")
    return data


def _index_by_id(responses: Sequence[dict]) -> dict:
    index = {}
    for entry in responses:
        response_id = entry.get("response")
        if not response_id:
            continue
        index[response_id] = entry
    return index


def _resolve_response_ids(response_ids: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(response_ids, str):
        return [response_ids]
    return list(response_ids)


def select_first(responses: Sequence[dict], count: int) -> List[dict]:
    if count <= 0:
        raise ValueError("count must be greater than zero")
    return list(responses[:count])


def select_random(responses: Sequence[dict], count: int, seed: Optional[int] = None) -> List[dict]:
    if count <= 0:
        raise ValueError("count must be greater than zero")
    if count > len(responses):
        raise ValueError("count cannot exceed the number of available responses")
    rng = random.Random(seed)
    return rng.sample(list(responses), count)


def select_by_ids(responses: Sequence[dict], response_ids: Union[str, Sequence[str]]) -> List[dict]:
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
    webhook = os.getenv(WEBHOOK_ENV_VAR)
    if not webhook:
        raise RuntimeError(f"Environment variable {WEBHOOK_ENV_VAR} is not set")
    return webhook


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _post_payload(payload: dict) -> List[dict]:
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
    saved_paths: List[Path] = []
    for judgement in judgements:
        response_id = judgement.get("response")
        if not response_id:
            logger.warning("Skipping webhook entry without response id: %s", judgement)
            continue
        target = output_dir / f"{response_id}_judged.json"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(judgement, handle, ensure_ascii=False, indent=2)
        saved_paths.append(target)
    return saved_paths


def _dispatch_entries(
    entries: Sequence[dict],
    *,
    output_dir: Path,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    _ensure_output_dir(output_dir)
    saved: List[Path] = []
    sender = post_fn or _post_payload
    for entry in entries:
        response_id = entry.get("response", "<unknown>")
        logger.info("Dispatching response %s", response_id)
        if dry_run:
            logger.info("Dry-run enabled, skipping webhook call for %s", response_id)
            continue
        judgements = sender(entry)
        saved.extend(_save_outputs(judgements, output_dir))
    return saved


def send_num_data_to_agents(
    count: int = DEFAULT_RANDOM_COUNT,
    *,
    randomize: bool = False,
    random_seed: Optional[int] = None,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    responses = load_responses(responses_path)
    if randomize:
        count = count or DEFAULT_RANDOM_COUNT
        entries = select_random(responses, count, seed=random_seed)
    else:
        entries = select_first(responses, count)
    return _dispatch_entries(entries, output_dir=output_dir, post_fn=post_fn, dry_run=dry_run)


def send_specific_responses(
    response_ids: Union[str, Sequence[str]],
    *,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    post_fn: Optional[PostFn] = None,
    dry_run: bool = False,
) -> List[Path]:
    responses = load_responses(responses_path)
    entries = select_by_ids(responses, response_ids)
    return _dispatch_entries(entries, output_dir=output_dir, post_fn=post_fn, dry_run=dry_run)


def main(
    *,
    count: int = DEFAULT_RANDOM_COUNT,
    randomize: bool = False,
    random_seed: Optional[int] = None,
    response_ids: Optional[Union[str, Sequence[str]]] = None,
    responses_path: Path = RAW_RESPONSES_PATH,
    output_dir: Path = OUTPUT_DIR,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> None:
    """Entry point callable without argparse.

    Adjust the keyword arguments when calling ``main`` to switch between modes.
    """

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if response_ids:
        saved = send_specific_responses(
            response_ids,
            responses_path=responses_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )
    else:
        saved = send_num_data_to_agents(
            count=count,
            randomize=randomize,
            random_seed=random_seed,
            responses_path=responses_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )

    logger.info("Saved %d webhook response files", len(saved))


if __name__ == "__main__":
    # Tweak these parameters for quick manual runs without argparse.
    main(response_ids=["3c5e25b6-2d4d-3d84-b01c-0264c3a5ba50","02693406-15df-33d5-b424-219ac8ab2054"])
