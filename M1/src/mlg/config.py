from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
DEFAULT_JUDGE_BASE_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_JUDGE_THINKING_MODE = "disabled"
DEFAULT_EMBEDDING_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model: str = os.getenv("LLM_MODEL", "qwen-plus-latest")
    judge_model: str = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    embedding_api_key: str = os.getenv(DEFAULT_EMBEDDING_API_KEY_ENV, "")
    embedding_api_key_env: str = os.getenv("LLM_EMBEDDING_API_KEY_ENV", DEFAULT_EMBEDDING_API_KEY_ENV)
    embedding_base_url: str = os.getenv("LLM_EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    embedding_model: str = os.getenv("LLM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_batch_max_items: int = int(os.getenv("LLM_EMBEDDING_BATCH_MAX_ITEMS", "50"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    context_window_tokens: int = int(os.getenv("LLM_CONTEXT_WINDOW_TOKENS", "1000000"))
    thinking_mode: str = os.getenv("LLM_THINKING_MODE", "").strip().lower()
    vector_cache_enabled: bool = os.getenv("MLG_VECTOR_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
    vector_cache_dir: str = os.getenv(
        "MLG_VECTOR_CACHE_DIR",
        str(DATA_DIR / "cache" / "vector_indexes"),
    )
    graph_cache_enabled: bool = os.getenv("MLG_GRAPH_CACHE_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    graph_cache_dir: str = os.getenv(
        "MLG_GRAPH_CACHE_DIR",
        str(DATA_DIR / "cache" / "taskgraphs"),
    )
    request_timeout_seconds: float = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "300"))
    long_context_min_start_interval_seconds: float = float(
        os.getenv("LLM_LONG_CONTEXT_MIN_START_INTERVAL_SECONDS", "150")
    )
    long_context_rate_limit_retries: int = int(os.getenv("LLM_LONG_CONTEXT_RATE_LIMIT_RETRIES", "2"))
    verify_ssl: bool = os.getenv("LLM_VERIFY_SSL", "false").lower() in {"1", "true", "yes"}
    ignore_proxy: bool = os.getenv("LLM_IGNORE_PROXY", "false").lower() in {"1", "true", "yes"}
    profile_name: str = "default"
    provider: str = "dashscope"

    @classmethod
    def from_profile(cls, profile: "ModelProfile", *, judge: bool = False) -> "RuntimeConfig":
        model_name = profile.model
        return cls(
            api_key=os.getenv(profile.api_key_env, ""),
            api_key_env=profile.api_key_env,
            base_url=profile.base_url,
            model=model_name,
            judge_model=model_name if judge else profile.judge_model,
            embedding_api_key=os.getenv(profile.embedding_api_key_env, ""),
            embedding_api_key_env=profile.embedding_api_key_env,
            embedding_base_url=profile.embedding_base_url,
            embedding_model=profile.embedding_model,
            embedding_batch_max_items=profile.embedding_batch_max_items,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            context_window_tokens=profile.context_window_tokens,
            thinking_mode=profile.thinking_mode,
            verify_ssl=profile.verify_ssl,
            ignore_proxy=profile.ignore_proxy,
            profile_name=profile.name,
            provider=profile.provider,
        )


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    base_url: str
    api_key_env: str
    model: str
    embedding_api_key_env: str = DEFAULT_EMBEDDING_API_KEY_ENV
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_max_items: int = 50
    judge_model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    context_window_tokens: int = 1_000_000
    thinking_mode: str = ""
    verify_ssl: bool = False
    ignore_proxy: bool = False


def default_model_profile(name: str = "default", model: str = "") -> ModelProfile:
    runtime = RuntimeConfig()
    return ModelProfile(
        name=name,
        provider=runtime.provider,
        base_url=runtime.base_url,
        api_key_env=runtime.api_key_env,
        model=model or runtime.model,
        embedding_api_key_env=runtime.embedding_api_key_env,
        embedding_base_url=runtime.embedding_base_url,
        embedding_model=runtime.embedding_model,
        embedding_batch_max_items=runtime.embedding_batch_max_items,
        judge_model=runtime.judge_model,
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        context_window_tokens=runtime.context_window_tokens,
        thinking_mode=runtime.thinking_mode,
        verify_ssl=runtime.verify_ssl,
        ignore_proxy=runtime.ignore_proxy,
    )


def load_model_profiles(path: Path | str) -> dict[str, ModelProfile]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    raw_profiles = payload.get("profiles", payload)
    if not isinstance(raw_profiles, dict):
        raise ValueError("model config must contain a 'profiles' object")
    profiles: dict[str, ModelProfile] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ValueError(f"model profile {name} must be an object")
        model = str(raw.get("model", "")).strip()
        if not model:
            raise ValueError(f"model profile {name} must define 'model'")
        api_key_env = str(raw.get("api_key_env", "")).strip()
        if not api_key_env:
            raise ValueError(f"model profile {name} must define 'api_key_env'")
        base_url = str(raw.get("base_url", "")) or RuntimeConfig().base_url
        profiles[name] = ModelProfile(
            name=name,
            provider=str(raw.get("provider", "")) or "openai-compatible",
            base_url=base_url,
            api_key_env=api_key_env,
            model=model,
            embedding_api_key_env=str(raw.get("embedding_api_key_env", "")) or RuntimeConfig().embedding_api_key_env,
            embedding_base_url=str(raw.get("embedding_base_url", "")) or RuntimeConfig().embedding_base_url or base_url,
            embedding_model=str(raw.get("embedding_model", "")) or RuntimeConfig().embedding_model,
            embedding_batch_max_items=int(
                raw.get("embedding_batch_max_items", RuntimeConfig().embedding_batch_max_items)
            ),
            judge_model=str(raw.get("judge_model", "")) or RuntimeConfig().judge_model,
            temperature=float(raw.get("temperature", RuntimeConfig().temperature)),
            max_tokens=int(raw.get("max_tokens", RuntimeConfig().max_tokens)),
            context_window_tokens=int(raw.get("context_window_tokens", RuntimeConfig().context_window_tokens)),
            thinking_mode=_thinking_mode(raw.get("thinking_mode", RuntimeConfig().thinking_mode)),
            verify_ssl=_as_bool(raw.get("verify_ssl", RuntimeConfig().verify_ssl)),
            ignore_proxy=_as_bool(raw.get("ignore_proxy", RuntimeConfig().ignore_proxy)),
        )
    return profiles


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _thinking_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode not in {"", "enabled", "disabled"}:
        raise ValueError("thinking_mode must be 'enabled', 'disabled', or empty")
    return mode


def select_model_profiles(
    *,
    model_config: Path | str | None = None,
    model_profiles: list[str] | None = None,
    fallback_model: str = "",
) -> dict[str, ModelProfile]:
    if model_config:
        available = load_model_profiles(model_config)
        selected_names = model_profiles or [name for name in available if "judge" not in name.lower()] or list(available)
        missing = [name for name in selected_names if name not in available]
        if missing:
            raise ValueError(f"Unknown model profile(s): {', '.join(missing)}")
        return {name: available[name] for name in selected_names}
    profile = default_model_profile(model=fallback_model)
    return {profile.name: profile}


def select_judge_profile(
    *,
    model_config: Path | str | None = None,
    judge_profile: str = "",
    fallback_judge_model: str = "",
) -> ModelProfile:
    if model_config:
        available = load_model_profiles(model_config)
        selected_name = judge_profile or ("fixed_judge" if "fixed_judge" in available else "")
        if selected_name:
            if selected_name not in available:
                raise ValueError(f"Unknown judge profile {selected_name}")
            return available[selected_name]
    runtime = RuntimeConfig()
    return ModelProfile(
        name=judge_profile or "default_judge",
        provider="deepseek",
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_JUDGE_BASE_URL),
        api_key_env=DEFAULT_JUDGE_API_KEY_ENV,
        model=fallback_judge_model or runtime.judge_model,
        embedding_api_key_env=runtime.embedding_api_key_env,
        embedding_base_url=runtime.embedding_base_url,
        embedding_model=runtime.embedding_model,
        judge_model=fallback_judge_model or runtime.judge_model,
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        context_window_tokens=runtime.context_window_tokens,
        thinking_mode=_thinking_mode(
            os.getenv("JUDGE_THINKING_MODE", DEFAULT_JUDGE_THINKING_MODE)
        ),
        verify_ssl=runtime.verify_ssl,
        ignore_proxy=runtime.ignore_proxy,
    )


def ensure_runtime_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
