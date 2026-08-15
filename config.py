"""Central configuration: model registry, credential loading, and run settings.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from dotenv import load_dotenv

# Load variables from a local .env file if present.
load_dotenv()

MODELS: dict[str, dict[str, Any]] = {
    "opus": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-opus-5",
        # Claude Opus 5 rejects the temperature param (400 if sent); its
        # always-on thinking counts against max_tokens, so give it headroom.
        "send_temperature": False,
        "max_tokens": 8192,
    },
    "sol": {
        "provider": "openrouter",
        "model_id": "openai/gpt-5.6-sol",
        # Reasoning model: rejects sampling params; same headroom logic.
        "send_temperature": False,
        "max_tokens": 8192,
    },
    "deepseek": {
        "provider": "azure",
        "model_id": "DeepSeek-V4-Pro",
        "endpoint_env": "AZURE_DEEPSEEK_ENDPOINT",
        "key_env": "AZURE_DEEPSEEK_API_KEY",
    },
    "grok": {
        "provider": "azure",
        "model_id": "grok-4-20-reasoning",
        "endpoint_env": "AZURE_GROK_ENDPOINT",
        "key_env": "AZURE_GROK_API_KEY",
    },
    "qwen": {
        "provider": "groq",
        "model_id": "qwen/qwen3.6-27b",
    },
}


MODEL_KEY: str = os.getenv("KHRA_MODEL") or "opus"
if MODEL_KEY not in MODELS:
    raise SystemExit(
        f"Unknown model key {MODEL_KEY!r}. Valid keys: {', '.join(sorted(MODELS))}"
    )

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY and (
    "put your api key" in GROQ_API_KEY.lower() or not GROQ_API_KEY.startswith("gsk_")
):
    GROQ_API_KEY = None


def _usable(value: str | None) -> bool:
    """True if an env value looks like a real credential, not a placeholder."""
    if not value:
        return False
    lowered = value.strip().lower()
    return bool(lowered) and "put_your" not in lowered and "put your" not in lowered


def has_credentials(spec: dict[str, Any]) -> bool:
    """True if .env carries usable credentials for this entry's provider."""
    provider = spec["provider"]
    if provider == "groq":
        return GROQ_API_KEY is not None
    if provider == "bedrock":
        # Either auth works for the native Bedrock (Claude) branch: a standard
        # IAM access-key pair, OR a Bedrock API key (ABSK..., set as
        # AWS_BEARER_TOKEN_BEDROCK — boto3 picks it up from the env natively).
        return (
            _usable(os.getenv("AWS_ACCESS_KEY_ID"))
            and _usable(os.getenv("AWS_SECRET_ACCESS_KEY"))
        ) or _usable(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
    if provider == "bedrock-openai":
        return _usable(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
    if provider == "azure":
        return _usable(os.getenv(spec["endpoint_env"])) and _usable(
            os.getenv(spec["key_env"])
        )
    if provider == "openrouter":
        return _usable(os.getenv("OPENROUTER_API_KEY"))
    return False


#mock mode
def _mock_mode_for(spec: dict[str, Any]) -> bool:
    mock_env = os.getenv("KHRA_MOCK")
    if mock_env is not None:
        return mock_env.strip().lower() in {"1", "true", "yes", "on"}
    return not has_credentials(spec)



MODEL_SPEC: dict[str, Any] = MODELS[MODEL_KEY]
PROVIDER: str = MODEL_SPEC["provider"]
MODEL_ID: str = MODEL_SPEC["model_id"]
MOCK_MODE: bool = _mock_mode_for(MODEL_SPEC)


def require_api_key() -> str:
    """Fail fast with a clear message if the Groq key is missing (groq path)."""
    if not GROQ_API_KEY:
        raise SystemExit(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "Groq API key (https://console.groq.com/keys), then rerun. "
            "To test the pipeline without a key, run with KHRA_MOCK=1."
        )
    return GROQ_API_KEY


def require_aws_region() -> str:
    """The Bedrock region, failing fast with a setup pointer if unset."""
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not _usable(region):
        raise SystemExit(
            "AWS_REGION is not set. Add it to .env alongside AWS_ACCESS_KEY_ID "
            "and AWS_SECRET_ACCESS_KEY (see .env.example). To test offline, "
            "run with KHRA_MOCK=1."
        )
    return region  


def require_bedrock_openai() -> tuple[str, str]:
    """(base_url, api_key) for Bedrock's OpenAI-compatible endpoint."""
    key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not _usable(key):
        raise SystemExit(
            "AWS_BEARER_TOKEN_BEDROCK is not set. Generate a Bedrock API key in "
            "the AWS console and add it to .env (see .env.example). To test "
            "offline, run with KHRA_MOCK=1."
        )
    base_url = os.getenv("BEDROCK_OPENAI_BASE_URL") or (
        f"https://bedrock-runtime.{require_aws_region()}.amazonaws.com/openai/v1"
    )
    return base_url, key


def require_openrouter() -> tuple[str, str]:
    """(base_url, api_key) for OpenRouter's OpenAI-compatible endpoint."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not _usable(key):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Add your OpenRouter key "
            "(https://openrouter.ai/keys) to .env (see .env.example). To test "
            "offline, run with KHRA_MOCK=1."
        )
    base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    return base_url, key  # type: ignore[return-value]


def require_azure(spec: dict[str, Any]) -> tuple[str, str]:
    endpoint = os.getenv(spec["endpoint_env"])
    key = os.getenv(spec["key_env"])
    if not (_usable(endpoint) and _usable(key)):
        raise SystemExit(
            f"{spec['endpoint_env']} / {spec['key_env']} not set. Deploy "
            f"{spec['model_id']!r} in Azure AI Foundry and add the endpoint URL "
            "and key to .env (see .env.example). To test offline, run with "
            "KHRA_MOCK=1."
        )
    return endpoint, key  # type: ignore[return-value]


#run settings 
MAX_TOKENS: int = 3072
LLM_TEMPERATURE: float = 0.0   # greedy decoding: reproducible, no averaging needed
REASONING_EFFORT: str | None = None
REASONING_FORMAT: str | None = "parsed"
FORCE_ANSWER_ON_TRUNCATION: bool = True
FORCE_ANSWER_MAX_TOKENS: int = 160   # small: the forced reply is answer-only
API_RETRIES: int = 5           # attempts on transient API/parse errors (429/5xx)
RETRY_BASE_DELAY: float = 2.0  # seconds; exponential backoff base
RETRY_MAX_DELAY: float = 60.0  # seconds; backoff ceiling
RETRY_JITTER: float = 0.5      # seconds; uniform jitter added to each sleep
REQUEST_PAUSE: float = 0.0     # optional fixed pause between calls (rate limiting)
TRUST_INIT: float = 1.0        # initial pairwise trust in Dynamic Teams
TRUST_STEP: float = 0.1        # trust increment/decrement per round
TRUST_MIN: float = 0.0         # trust clamp lower bound
TRUST_MAX: float = 2.0         # trust clamp upper bound
DATA_DIR: str = "data"         # MINT dataset JSON cached here


#logs 
LOG_ROOT: str = "logs"
RESULTS_ROOT: str = "results"

# Which multi-agent system(s) the run covers (e.g. "hier", "adv-base-dyn"),
# set by run_experiment via set_model(). Baked into every run-folder name so
# logs/results are self-describing: <date>_<model>_<arch-tag>.
ARCH_TAG: str | None = None


def _run_tag() -> str:
    base = "mock" if MOCK_MODE else MODEL_KEY
    return f"{base}_{ARCH_TAG}" if ARCH_TAG else base


RUN_TAG: str = _run_tag()

_RUN_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+?)(?:_(\d+))?$")


def _existing_runs(model_key: str) -> list[tuple[str, int, str]]:
    """Sorted ``(date, suffix, folder_name)`` for this model's run folders."""
    runs: list[tuple[str, int, str]] = []
    if os.path.isdir(LOG_ROOT):
        for name in os.listdir(LOG_ROOT):
            m = _RUN_NAME_RE.match(name)
            if (
                m
                and m.group(2) == model_key
                and os.path.isdir(os.path.join(LOG_ROOT, name))
            ):
                runs.append((m.group(1), int(m.group(3) or 1), name))
    return sorted(runs)


def _new_run_name(model_key: str) -> str:
    """Today's folder name, suffixed _2, _3, ... if one already exists."""
    today = time.strftime("%Y-%m-%d")
    name = f"{today}_{model_key}"
    n = 1
    while os.path.exists(os.path.join(LOG_ROOT, name)) or os.path.exists(
        os.path.join(RESULTS_ROOT, name)
    ):
        n += 1
        name = f"{today}_{model_key}_{n}"
    return name


def _resolve_run_name(model_key: str) -> str:
    """Resume the most recent run folder for this model, else name a fresh one."""
    runs = _existing_runs(model_key)
    return runs[-1][2] if runs else _new_run_name(model_key)


RUN_NAME: str = os.getenv("KHRA_RUN") or _resolve_run_name(RUN_TAG)
LOG_DIR: str = os.path.join(LOG_ROOT, RUN_NAME)          # this run's transcripts
RESULTS_DIR: str = os.path.join(RESULTS_ROOT, RUN_NAME)  # this run's result tables


def set_run(name: str) -> str:
    """Point LOG_DIR/RESULTS_DIR at a specific run folder (the --run flag)."""
    global RUN_NAME, LOG_DIR, RESULTS_DIR
    RUN_NAME = name
    LOG_DIR = os.path.join(LOG_ROOT, name)
    RESULTS_DIR = os.path.join(RESULTS_ROOT, name)
    return RUN_NAME


def start_new_run() -> str:
    """Switch to a fresh dated run folder (the --new-run flag)."""
    return set_run(_new_run_name(RUN_TAG))


def set_model(key: str, arch_tag: str | None = None) -> str:
    """Activate a registry entry (the --model flag): model, provider, mock
    mode, and the run folders all follow. ``arch_tag`` names the multi-agent
    system(s) being run (see architectures.arch_tag) and becomes part of the
    run-folder name. Returns the resolved run name."""
    global MODEL_KEY, MODEL_SPEC, PROVIDER, MODEL_ID, MOCK_MODE, RUN_TAG, ARCH_TAG
    if key not in MODELS:
        raise SystemExit(
            f"Unknown model key {key!r}. Valid keys: {', '.join(sorted(MODELS))}"
        )
    MODEL_KEY = key
    MODEL_SPEC = MODELS[key]
    PROVIDER = MODEL_SPEC["provider"]
    MODEL_ID = MODEL_SPEC["model_id"]
    MOCK_MODE = _mock_mode_for(MODEL_SPEC)
    ARCH_TAG = arch_tag
    RUN_TAG = _run_tag()
    return set_run(os.getenv("KHRA_RUN") or _resolve_run_name(RUN_TAG))




MINT_STRATEGY: str = "other"   # misinformation strategy; swappable for ablations
RANDOM_SEED: int = 42          # option-shuffle seed ONLY (no question sampling)


ACTIVE_CONDITIONS: list[str] = ["control", "single"]

DAILY_TOKEN_CAP: int = 200_000   # previous daily cap for groq free tier
TOKEN_WINDOW_HOURS: float = 24.0

THINKING_OFF: bool = False
NO_THINK_DIRECTIVE: str = "/no_think"

ACTIVE_CATEGORIES: list[str] = ["winogrande"]
