"""
title: NOMYO Router - an Ollama Proxy with Endpoint:Model aware routing
author: alpha-nerd-nomyo
author_url: https://github.com/nomyo-ai
version: 0.3
license: AGPL
"""
# -------------------------------------------------------------
import json, time, asyncio, yaml, ollama, openai, os, re, aiohttp, ssl, datetime, random
from pathlib import Path
from typing import Dict, Set, List, Optional, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi_sse import sse_handler
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse, Response, HTMLResponse, RedirectResponse
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from collections import defaultdict
from dotenv import load_dotenv

# HuggingFace integration (optional)
try:
    from huggingface_hub import InferenceClient, HfApi
    HF_AVAILABLE = True
    # ModelFilter is optional, only needed for model discovery
    try:
        from huggingface_hub import ModelFilter
    except ImportError:
        ModelFilter = None
except ImportError:
    HF_AVAILABLE = False
    InferenceClient = None
    HfApi = None
    ModelFilter = None

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------------
# In‑memory caches
# ------------------------------------------------------------------
# Successful results are cached for 300s
_models_cache: dict[str, tuple[Set[str], float]] = {}
# Transient errors are cached for 1s – the key stays until the
# timeout expires, after which the endpoint will be queried again.
_error_cache: dict[str, float] = {}

# Hardware stats cache
_hardware_cache: Dict[str, any] = {
    "stats": None,
    "cached_at": 0,
    "ttl": 60  # Cache for 60 seconds
}

# ------------------------------------------------------------------
# SSE Queues
# ------------------------------------------------------------------
_subscribers: Set[asyncio.Queue] = set()
_subscribers_lock = asyncio.Lock()

# ------------------------------------------------------------------
# aiohttp Global Sessions
# ------------------------------------------------------------------
app_state = {
    "session": None,
    "connector": None,
}

# -------------------------------------------------------------
# 1. Configuration loader
# -------------------------------------------------------------
class Config(BaseSettings):
    # List of endpoints (can be string or dict with provider field)
    endpoints: list[Union[str, dict]] = Field(
        default_factory=lambda: [
            "http://localhost:11434",
        ]
    )
    # Max concurrent connections per endpoint‑model pair, see OLLAMA_NUM_PARALLEL
    max_concurrent_connections: int = 1

    api_keys: Dict[str, str] = Field(default_factory=dict)

    # Internal mapping for provider information
    endpoint_info: Dict[str, Dict[str, str]] = Field(default_factory=dict)

    class Config:
        # Load from `config.yaml` first, then from env variables
        env_prefix = "NOMYO_ROUTER_"
        yaml_file = Path("config.yaml")  # relative to cwd

    @classmethod
    def _expand_env_refs(cls, obj):
        """Recursively replace `${VAR}` with os.getenv('VAR')."""
        if isinstance(obj, dict):
            return {k: cls._expand_env_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._expand_env_refs(v) for v in obj]
        if isinstance(obj, str):
            # Only expand if it is exactly ${VAR}
            m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", obj)
            if m:
                return os.getenv(m.group(1), "")
        return obj

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load the YAML file and create the Config instance."""
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
                cleaned = cls._expand_env_refs(data)
            config = cls(**cleaned)
            config._normalize_endpoints()
            return config
        config = cls()
        config._normalize_endpoints()
        return config

    def _normalize_endpoints(self):
        """Convert mixed endpoint formats to internal representation."""
        normalized = []
        for ep in self.endpoints:
            if isinstance(ep, str):
                # Auto-detect from URL
                provider = "openai" if "/v1" in ep else "ollama"
                url = ep
            elif isinstance(ep, dict):
                url = ep.get("url")
                provider = ep.get("provider", "auto")
                if provider == "auto":
                    provider = "openai" if "/v1" in url else "ollama"
            else:
                continue  # Skip invalid entries

            normalized.append(url)
            self.endpoint_info[url] = {
                "provider": provider,
                "url": url
            }

        self.endpoints = normalized

    def get_provider(self, endpoint: str) -> str:
        """Get provider type for an endpoint."""
        return self.endpoint_info.get(endpoint, {}).get("provider", "ollama")

# Create the global config object – it will be overwritten on startup
config = Config()

# -------------------------------------------------------------
# 2. FastAPI application
# -------------------------------------------------------------
app = FastAPI()
sse_handler.app = app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
default_headers={
            "HTTP-Referer": "https://nomyo.ai",
            "X-Title": "NOMYO Router",
            }
        
# -------------------------------------------------------------
# 3. Global state: per‑endpoint per‑model active connection counters
# -------------------------------------------------------------
usage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
usage_lock = asyncio.Lock()  # protects access to usage_counts

# -------------------------------------------------------------
# 4. Helperfunctions 
# -------------------------------------------------------------
aiotimeout = aiohttp.ClientTimeout(total=5)

def _is_fresh(cached_at: float, ttl: int) -> bool:
    return (time.time() - cached_at) < ttl

async def _ensure_success(resp: aiohttp.ClientResponse) -> None:
    if resp.status >= 400:
        text = await resp.text()
        raise HTTPException(status_code=resp.status, detail=text)

class fetch:
    async def available_models(endpoint: str, api_key: Optional[str] = None) -> Set[str]:
        """
        Query <endpoint>/api/tags and return a set of all model names that the
        endpoint *advertises* (i.e. is capable of serving).  This endpoint lists
        every model that is installed on the Ollama instance, regardless of
        whether the model is currently loaded into memory.

        If the request fails (e.g. timeout, 5xx, or malformed response), an empty
        set is returned.
        """
        headers = None
        if api_key is not None:
            headers = {"Authorization": "Bearer " + api_key}

        if endpoint in _models_cache:
            models, cached_at = _models_cache[endpoint]
            if _is_fresh(cached_at, 300):
                return models
            else:
                # stale entry – drop it
                del _models_cache[endpoint]

        if endpoint in _error_cache:
            if _is_fresh(_error_cache[endpoint], 1):
                # Still within the short error TTL – pretend nothing is available
                return set()
            else:
                # Error expired – remove it
                del _error_cache[endpoint]

        provider = get_provider(endpoint)

        if provider == "huggingface":
            # HF Inference API - no global model list
            # Models are checked at request time
            return set()

        elif provider in ["openai", "tgi"]:
            # OpenAI-compatible endpoint (includes TGI)
            endpoint_url = f"{endpoint}/models"
            key = "data"
        else:
            # Ollama endpoint
            endpoint_url = f"{endpoint}/api/tags"
            key = "models"

        client: aiohttp.ClientSession = app_state["session"]
        try:
            async with client.get(endpoint_url, headers=headers) as resp:
                await _ensure_success(resp)
                data = await resp.json()

                items = data.get(key, [])
                models = {item.get("id") or item.get("name") for item in items if item.get("id") or item.get("name")}
                
                if models:
                    _models_cache[endpoint] = (models, time.time())
                    return models
                else:
                    # Empty list – treat as “no models”, but still cache for 300s
                    _models_cache[endpoint] = (models, time.time())
                    return models
        except Exception as e:
            # Treat any error as if the endpoint offers no models
            print(f"[fetch.available_models] {endpoint} error: {e}")
            _error_cache[endpoint] = time.time()
            return set()


    async def loaded_models(endpoint: str) -> Set[str]:
        """
        Query <endpoint>/api/ps and return a set of model names that are currently
        loaded on that endpoint. If the request fails (e.g. timeout, 5xx), an empty
        set is returned.
        """
        provider = get_provider(endpoint)

        # Only Ollama supports /api/ps
        if provider != "ollama":
            return set()

        client: aiohttp.ClientSession = app_state["session"]
        try:
            async with client.get(f"{endpoint}/api/ps") as resp:
                await _ensure_success(resp)
                data = await resp.json()
            # The response format is:
            #   {"models": [{"name": "model1"}, {"name": "model2"}]}
            models = {m.get("name") for m in data.get("models", []) if m.get("name")}
            return models
        except Exception:
            # If anything goes wrong we simply assume the endpoint has no models
            return set()

    async def endpoint_details(endpoint: str, route: str, detail: str, api_key: Optional[str] = None) -> List[dict]:
        """
        Query <endpoint>/<route> to fetch <detail> and return a List of dicts with details
        for the corresponding Ollama endpoint. If the request fails we respond with "N/A" for detail.
        """
        client: aiohttp.ClientSession = app_state["session"]
        headers = None
        if api_key is not None:
            headers = {"Authorization": "Bearer " + api_key}
        
        try:
            async with client.get(f"{endpoint}{route}", headers=headers) as resp:
                await _ensure_success(resp)
                data = await resp.json()
            detail = data.get(detail, [])
            return detail
        except Exception as e:
            # If anything goes wrong we cannot reply details
            print(e)
            return []

def ep2base(ep):
    if "/v1" in ep:
        base_url = ep
    else:
        base_url = ep+"/v1"
    return base_url

def get_provider(endpoint: str) -> str:
    """Get provider type for an endpoint."""
    return config.get_provider(endpoint)

def dedupe_on_keys(dicts, key_fields):
    """
    Helper function to deduplicate endpoint details based on given dict keys.
    """
    seen = set()
    out = []
    for d in dicts:
        # Build a tuple of the values for the chosen keys
        key = tuple(d.get(k) for k in key_fields)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

async def increment_usage(endpoint: str, model: str) -> None:
    async with usage_lock:
        usage_counts[endpoint][model] += 1
    await publish_snapshot()

async def decrement_usage(endpoint: str, model: str) -> None:
    async with usage_lock:
        # Avoid negative counts
        current = usage_counts[endpoint].get(model, 0)
        if current > 0:
            usage_counts[endpoint][model] = current - 1
        # Optionally, clean up zero entries
        if usage_counts[endpoint].get(model, 0) == 0:
            usage_counts[endpoint].pop(model, None)
        #if not usage_counts[endpoint]:
        #    usage_counts.pop(endpoint, None)
    await publish_snapshot()

def iso8601_ns():
    ns_since_epoch = time.time_ns()
    dt = datetime.datetime.fromtimestamp(
        ns_since_epoch / 1_000_000_000,  # seconds
        tz=datetime.timezone.utc
    )
    iso8601_with_ns = (
        dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ns_since_epoch % 1_000_000_000:09d}Z"
    )
    return iso8601_with_ns

class rechunk:
    def openai_chat_completion2ollama(chunk: dict, stream: bool, start_ts: float) -> ollama.ChatResponse:
        if chunk.choices == [] and chunk.usage is not None:
            return ollama.ChatResponse(
                model=chunk.model,
                created_at=iso8601_ns(),
                done=True,
                done_reason='stop',
                total_duration=int((time.perf_counter() - start_ts) * 1_000_000_000),
                load_duration=100000,
                prompt_eval_count=int(chunk.usage.prompt_tokens),
                prompt_eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)) if chunk.usage.completion_tokens != 0 else 0,
                eval_count=int(chunk.usage.completion_tokens),
                eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000),
                message={"role": "assistant"}
                )
        with_thinking = chunk.choices[0] if chunk.choices[0] else None
        if stream == True:
            thinking = getattr(with_thinking.delta, "reasoning", None) if with_thinking else None
            role = chunk.choices[0].delta.role or "assistant"
            content = chunk.choices[0].delta.content or ''
        else:
            thinking = getattr(with_thinking, "reasoning", None) if with_thinking else None
            role = chunk.choices[0].message.role or "assistant"
            content = chunk.choices[0].message.content or ''
        assistant_msg = ollama.Message(
            role=role,
            content=content,
            thinking=thinking,
            images=None,
            tool_name=None,
            tool_calls=None)
        rechunk = ollama.ChatResponse(
            model=chunk.model, 
            created_at=iso8601_ns(),
            done=True if chunk.usage is not None else False,
            done_reason=chunk.choices[0].finish_reason, #if chunk.choices[0].finish_reason is not None else None,
            total_duration=int((time.perf_counter() - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            load_duration=100000, 
            prompt_eval_count=int(chunk.usage.prompt_tokens) if chunk.usage is not None else 0,
            prompt_eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)) if chunk.usage is not None and chunk.usage.completion_tokens != 0 else 0, 
            eval_count=int(chunk.usage.completion_tokens) if chunk.usage is not None else 0,
            eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            message=assistant_msg)
        return rechunk
    
    def openai_completion2ollama(chunk: dict, stream: bool, start_ts: float) -> ollama.GenerateResponse:
        with_thinking = chunk.choices[0] if chunk.choices[0] else None
        thinking = getattr(with_thinking, "reasoning", None) if with_thinking else None
        rechunk = ollama.GenerateResponse(
            model=chunk.model,
            created_at=iso8601_ns(),
            done=True if chunk.usage is not None else False,
            done_reason=chunk.choices[0].finish_reason,
            total_duration=int((time.perf_counter() - start_ts) * 1000) if chunk.usage is not None else 0,
            load_duration=10000,
            prompt_eval_count=int(chunk.usage.prompt_tokens) if chunk.usage is not None else 0,
            prompt_eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)) if chunk.usage is not None and chunk.usage.completion_tokens != 0 else 0,
            eval_count=int(chunk.usage.completion_tokens) if chunk.usage is not None else 0,
            eval_duration=int((time.perf_counter() - start_ts) * 1000) if chunk.usage is not None else 0,
            response=chunk.choices[0].text or '',
            thinking=thinking)
        return rechunk
    
    def openai_embeddings2ollama(chunk: dict) -> ollama.EmbeddingsResponse:
        rechunk = ollama.EmbeddingsResponse(embedding=chunk.data[0].embedding)
        return rechunk

    def openai_embed2ollama(chunk: dict, model: str) -> ollama.EmbedResponse:
        rechunk = ollama.EmbedResponse(
            model=model,
            created_at=iso8601_ns(),
            done=None,
            done_reason=None,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None,
            embeddings=[chunk.data[0].embedding])
        return rechunk

    def hf_chat2ollama(chunk: dict, stream: bool, start_ts: float, model: str) -> ollama.ChatResponse:
        """Convert HuggingFace text-generation to Ollama chat format."""
        if stream:
            # Streaming: {"token": {"text": "..."}, ...}
            token_text = chunk.get("token", {}).get("text", "")
            is_done = chunk.get("generated_text") is not None

            return ollama.ChatResponse(
                model=model,
                created_at=iso8601_ns(),
                done=is_done,
                done_reason='stop' if is_done else None,
                total_duration=int((time.perf_counter() - start_ts) * 1_000_000_000) if is_done else 0,
                load_duration=100000,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000) if is_done else 0,
                message=ollama.Message(
                    role="assistant",
                    content=token_text,
                    thinking=None,
                    images=None,
                    tool_name=None,
                    tool_calls=None
                )
            )
        else:
            # Non-streaming: [{"generated_text": "..."}]
            generated_text = chunk[0].get("generated_text", "") if chunk else ""

            return ollama.ChatResponse(
                model=model,
                created_at=iso8601_ns(),
                done=True,
                done_reason='stop',
                total_duration=int((time.perf_counter() - start_ts) * 1_000_000_000),
                load_duration=100000,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000),
                message=ollama.Message(
                    role="assistant",
                    content=generated_text,
                    thinking=None,
                    images=None,
                    tool_name=None,
                    tool_calls=None
                )
            )

    def hf_generate2ollama(chunk: dict, stream: bool, start_ts: float, model: str) -> ollama.GenerateResponse:
        """Convert HuggingFace to Ollama generate format."""
        if stream:
            token_text = chunk.get("token", {}).get("text", "")
            is_done = chunk.get("generated_text") is not None

            return ollama.GenerateResponse(
                model=model,
                created_at=iso8601_ns(),
                done=is_done,
                done_reason='stop' if is_done else None,
                total_duration=int((time.perf_counter() - start_ts) * 1000) if is_done else 0,
                load_duration=10000,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=int((time.perf_counter() - start_ts) * 1000) if is_done else 0,
                response=token_text,
                thinking=None
            )
        else:
            generated_text = chunk[0].get("generated_text", "") if chunk else ""

            return ollama.GenerateResponse(
                model=model,
                created_at=iso8601_ns(),
                done=True,
                done_reason='stop',
                total_duration=int((time.perf_counter() - start_ts) * 1000),
                load_duration=10000,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=int((time.perf_counter() - start_ts) * 1000),
                response=generated_text,
                thinking=None
            )

    def hf_embeddings2ollama(embeddings: list, model: str) -> ollama.EmbedResponse:
        """Convert HuggingFace embeddings to Ollama format."""
        return ollama.EmbedResponse(
            model=model,
            created_at=iso8601_ns(),
            done=True,
            done_reason=None,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None,
            embeddings=[embeddings]
        )

def messages_to_prompt(messages: list) -> str:
    """Convert Ollama/OpenAI messages to plain prompt for HF."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)

# ------------------------------------------------------------------
# Hardware Detection & Compatibility Check
# ------------------------------------------------------------------
class HardwareStats(BaseModel):
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    disk_total_gb: float
    disk_available_gb: float
    disk_used_percent: float
    cpu_cores: int
    gpu_available: bool
    platform: str

async def get_hardware_stats() -> HardwareStats:
    """Get system hardware stats with caching."""
    import time

    # Check cache
    if (_hardware_cache["stats"] is not None and
        time.time() - _hardware_cache["cached_at"] < _hardware_cache["ttl"]):
        return _hardware_cache["stats"]

    # Read /proc/meminfo for RAM
    with open("/proc/meminfo", "r") as f:
        meminfo = f.read()
        mem_total_kb = int([l for l in meminfo.split("\n") if "MemTotal" in l][0].split()[1])
        mem_available_kb = int([l for l in meminfo.split("\n") if "MemAvailable" in l][0].split()[1])

    # Get disk stats
    import os
    stat = os.statvfs("/")
    disk_total_bytes = stat.f_blocks * stat.f_frsize
    disk_available_bytes = stat.f_bavail * stat.f_frsize

    # Build stats object
    stats = HardwareStats(
        ram_total_gb=round(mem_total_kb / 1024 / 1024, 1),
        ram_available_gb=round(mem_available_kb / 1024 / 1024, 1),
        ram_used_percent=round((1 - mem_available_kb / mem_total_kb) * 100),
        disk_total_gb=round(disk_total_bytes / 1024 / 1024 / 1024, 1),
        disk_available_gb=round(disk_available_bytes / 1024 / 1024 / 1024, 2),
        disk_used_percent=round((1 - stat.f_bavail / stat.f_blocks) * 100),
        cpu_cores=os.cpu_count() or 0,
        gpu_available=False,  # No GPU on Android/Termux
        platform="android"
    )

    # Cache and return
    _hardware_cache["stats"] = stats
    _hardware_cache["cached_at"] = time.time()

    return stats

def estimate_ollama_model_size(model_name: str) -> float:
    """Estimate model size from name (e.g., 'llama3:8b' -> 4.7GB)."""
    size_map = {
        "2b": {"q4_0": 1.4, "q4_k_m": 1.5, "q5_0": 1.6},
        "3b": {"q4_0": 2.0, "q4_k_m": 2.1},
        "7b": {"q4_0": 3.8, "q4_k_m": 4.1, "q5_0": 4.5},
        "8b": {"q4_0": 4.7, "q4_k_m": 5.0},
        "13b": {"q4_0": 7.4, "q4_k_m": 7.9},
        "70b": {"q4_0": 39.0, "q4_k_m": 42.0}
    }

    # Parse model name
    name_lower = model_name.lower()

    # Find parameter size (2b, 3b, 7b, etc.)
    param_size = None
    for size in size_map.keys():
        if size in name_lower:
            param_size = size
            break

    if not param_size:
        return 5.0  # Default: assume 7b model

    # Assume Q4_0 quantization if not specified
    return size_map[param_size].get("q4_0", 5.0)

async def check_model_compatibility(model_name: str, provider: str) -> dict:
    """Check if model is compatible with current hardware."""
    hw = await get_hardware_stats()

    if provider == "huggingface":
        # HF Inference API - no local requirements
        return {
            "compatible": True,
            "status": "ok",
            "checks": {
                "disk_space": {"status": "ok", "message": "No local storage needed"},
                "ram": {"status": "ok", "message": "Runs on HF servers"},
                "cpu": {"status": "ok", "message": "Runs on HF servers"}
            },
            "recommendations": []
        }

    # Ollama model - estimate size and check
    model_size_gb = estimate_ollama_model_size(model_name)

    checks = {}
    recommendations = []
    compatible = True
    overall_status = "ok"

    # Disk space check
    required_disk = model_size_gb + 0.5  # Model + 500MB buffer
    if hw.disk_available_gb < required_disk:
        checks["disk_space"] = {
            "required_gb": model_size_gb,
            "available_gb": hw.disk_available_gb,
            "status": "error",
            "message": f"Insufficient disk space. Need {model_size_gb}GB, have {hw.disk_available_gb}GB"
        }
        compatible = False
        overall_status = "error"
        recommendations.append(f"Free up at least {required_disk:.1f}GB disk space")
    elif hw.disk_available_gb < (model_size_gb + 2.0):
        checks["disk_space"] = {
            "required_gb": model_size_gb,
            "available_gb": hw.disk_available_gb,
            "status": "warning",
            "message": f"Low disk space. Only {hw.disk_available_gb}GB available"
        }
        if overall_status == "ok":
            overall_status = "warning"
    else:
        checks["disk_space"] = {
            "status": "ok",
            "message": f"{hw.disk_available_gb}GB available"
        }

    # RAM check
    required_ram = model_size_gb
    if hw.ram_available_gb < required_ram:
        checks["ram"] = {
            "required_gb": required_ram,
            "available_gb": hw.ram_available_gb,
            "status": "error",
            "message": f"Insufficient RAM. Need {required_ram}GB, have {hw.ram_available_gb}GB"
        }
        compatible = False
        overall_status = "error"
        recommendations.append(f"Need at least {required_ram}GB RAM")
    elif hw.ram_available_gb < (required_ram * 1.5):
        checks["ram"] = {
            "required_gb": required_ram,
            "available_gb": hw.ram_available_gb,
            "status": "warning",
            "message": "RAM borderline for large contexts"
        }
        if overall_status == "ok":
            overall_status = "warning"
    else:
        checks["ram"] = {
            "status": "ok",
            "message": f"{hw.ram_available_gb}GB available"
        }

    # CPU check
    if hw.cpu_cores < 4:
        checks["cpu"] = {
            "status": "warning",
            "message": f"Only {hw.cpu_cores} cores - inference may be slow"
        }
        if overall_status == "ok":
            overall_status = "warning"
    else:
        checks["cpu"] = {
            "status": "ok",
            "message": f"{hw.cpu_cores} cores sufficient"
        }

    # Add recommendations for smaller models if incompatible
    if not compatible:
        if model_size_gb > 4.0:
            recommendations.append("Consider phi3:mini (2.3GB) or gemma:2b (1.4GB) instead")
        elif model_size_gb > 2.0:
            recommendations.append("Consider gemma:2b (1.4GB) instead")

    return {
        "compatible": compatible,
        "status": overall_status,
        "checks": checks,
        "recommendations": recommendations
    }

# ------------------------------------------------------------------
# SSE Helpser
# ------------------------------------------------------------------
async def publish_snapshot():
    async with usage_lock:
        snapshot = json.dumps({"usage_counts": usage_counts}, sort_keys=True)
    async with _subscribers_lock:
        for q in _subscribers:
            # If the queue is full, drop the message to avoid back‑pressure.
            if q.full():
                try:
                    await q.get()
                except asyncio.QueueEmpty:
                    pass
            await q.put(snapshot)

async def close_all_sse_queues():
    for q in list(_subscribers):
        # sentinel value that the generator will recognise
        await q.put(None)

# ------------------------------------------------------------------
# Subscriber helpers
# ------------------------------------------------------------------
async def subscribe() -> asyncio.Queue:
    """
    Returns a new Queue that will receive every snapshot.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    async with _subscribers_lock:
        _subscribers.add(q)
    return q

async def unsubscribe(q: asyncio.Queue):
    async with _subscribers_lock:
        _subscribers.discard(q)

# ------------------------------------------------------------------
# Convenience wrapper – returns the current snapshot (for the proxy)
# ------------------------------------------------------------------
async def get_usage_counts() -> Dict:
    return dict(usage_counts)   # shallow copy

# -------------------------------------------------------------
# 5. Endpoint selection logic (respecting the configurable limit)
# -------------------------------------------------------------
async def choose_endpoint(model: str) -> str:
    """
    Determine which endpoint to use for the given model while respecting
    the `max_concurrent_connections` per endpoint‑model pair **and**
    ensuring that the chosen endpoint actually *advertises* the model.

    The selection algorithm:

    1️⃣  Query every endpoint for its advertised models (`/api/tags`).
    2️⃣  Build a list of endpoints that contain the requested model.
    3️⃣  For those endpoints, find those that have the model loaded
        (`/api/ps`) *and* still have a free slot.
    4️⃣  If none are both loaded and free, fall back to any endpoint
        from the filtered list that simply has a free slot and randomly 
        select one.
    5️⃣  If all are saturated, pick any endpoint from the filtered list
        (the request will queue on that endpoint).
    6️⃣  If no endpoint advertises the model at all, raise an error.
    """
    # 1️⃣  Gather advertised‑model sets for all endpoints concurrently
    tag_tasks = [fetch.available_models(ep) for ep in config.endpoints if "/v1" not in ep]
    tag_tasks += [fetch.available_models(ep, config.api_keys[ep]) for ep in config.endpoints if "/v1" in ep]
    advertised_sets = await asyncio.gather(*tag_tasks)

    # 2️⃣  Filter endpoints that advertise the requested model
    candidate_endpoints = [
        ep for ep, models in zip(config.endpoints, advertised_sets)
        if model in models
    ]
    
    # 6️⃣ 
    if not candidate_endpoints:
        if ":latest" in model:  #ollama naming convention not applicable to openai
            model = model.split(":latest")
            model = model[0]
            candidate_endpoints = [
                ep for ep, models in zip(config.endpoints, advertised_sets)
                if model in models
            ]
        if not candidate_endpoints:
            raise RuntimeError(
                f"None of the configured endpoints ({', '.join(config.endpoints)}) "
                f"advertise the model '{model}'."
            )

    # 3️⃣  Among the candidates, find those that have the model *loaded*
    #      (concurrently, but only for the filtered list)
    load_tasks = [fetch.loaded_models(ep) for ep in candidate_endpoints]
    loaded_sets = await asyncio.gather(*load_tasks)
    
    async with usage_lock:
        # Helper: get current usage count for (endpoint, model)
        def current_usage(ep: str) -> int:
            return usage_counts.get(ep, {}).get(model, 0)
        
        # 3️⃣ Endpoints that have the model loaded *and* a free slot
        loaded_and_free = [
            ep for ep, models in zip(candidate_endpoints, loaded_sets)
            if model in models and usage_counts.get(ep, {}).get(model, 0) < config.max_concurrent_connections
        ]
        
        if loaded_and_free:
            ep = min(loaded_and_free, key=current_usage)
            return ep

        # 4️⃣ Endpoints among the candidates that simply have a free slot
        endpoints_with_free_slot = [
            ep for ep in candidate_endpoints
            if usage_counts.get(ep, {}).get(model, 0) < config.max_concurrent_connections
        ]

        if endpoints_with_free_slot:
            return random.choice(endpoints_with_free_slot)

        # 5️⃣ All candidate endpoints are saturated – pick one with lowest usages count (will queue)
        ep = min(candidate_endpoints, key=current_usage)
        return ep

# -------------------------------------------------------------
# 6. API route – Generate
# -------------------------------------------------------------
@app.post("/api/generate")
async def proxy(request: Request):
    """
    Proxy a generate request to Ollama and stream the response back to the client.
    """
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))
        
        model = payload.get("model")
        prompt = payload.get("prompt")
        suffix = payload.get("suffix")
        system = payload.get("system")
        template = payload.get("template")
        context = payload.get("context")
        stream = payload.get("stream")
        think = payload.get("think")
        raw = payload.get("raw")
        _format = payload.get("format")
        images = payload.get("images")
        options = payload.get("options")
        keep_alive = payload.get("keep_alive")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Missing required field 'prompt'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e


    endpoint = await choose_endpoint(model)
    provider = get_provider(endpoint)

    hf_client = None
    oclient = None
    client = None

    if provider == "huggingface":
        if not HF_AVAILABLE:
            raise HTTPException(status_code=500, detail="HuggingFace support not available. Install huggingface_hub.")

        # HuggingFace Inference API
        api_key = config.api_keys.get(endpoint, "")
        hf_client = InferenceClient(token=api_key)

        params = {
            "model": model,
            "prompt": prompt,
            "stream": stream if stream is not None else False,
        }

        # Optional parameters
        if options:
            if "temperature" in options:
                params["temperature"] = options["temperature"]
            if "num_predict" in options:
                params["max_new_tokens"] = options["num_predict"]
            if "top_p" in options:
                params["top_p"] = options["top_p"]

    elif provider in ["openai", "tgi"]:
        if ":latest" in model:
            model = model.split(":latest")[0]

        params = {
            "prompt": prompt,
            "model": model,
        }

        optional_params = {
            "stream": stream,
            "max_tokens": options.get("num_predict") if options and "num_predict" in options else None,
            "frequency_penalty": options.get("frequency_penalty") if options and "frequency_penalty" in options else None,
            "presence_penalty": options.get("presence_penalty") if options and "presence_penalty" in options else None,
            "seed": options.get("seed") if options and "seed" in options else None,
            "stop": options.get("stop") if options and "stop" in options else None,
            "top_p": options.get("top_p") if options and "top_p" in options else None,
            "temperature": options.get("temperature") if options and "temperature" in options else None,
            "suffix": suffix,
        }
        params.update({k: v for k, v in optional_params.items() if v is not None})
        oclient = openai.AsyncOpenAI(base_url=endpoint, default_headers=default_headers, api_key=config.api_keys[endpoint])

    else:  # ollama
        client = ollama.AsyncClient(host=endpoint)

    await increment_usage(endpoint, model)

    # 4. Async generator that streams data and decrements the counter
    async def stream_generate_response():
        try:
            if provider == "huggingface":
                start_ts = time.perf_counter()
                async_gen = hf_client.text_generation(**params, details=True)
            elif provider in ["openai", "tgi"]:
                start_ts = time.perf_counter()
                async_gen = await oclient.completions.create(**params)
            else:  # ollama
                async_gen = await client.generate(model=model, prompt=prompt, suffix=suffix, system=system, template=template, context=context, stream=stream, think=think, raw=raw, format=_format, images=images, options=options, keep_alive=keep_alive)

            if stream:
                async for chunk in async_gen:
                    if provider == "huggingface":
                        chunk = rechunk.hf_generate2ollama(chunk, stream, start_ts, model)
                    elif provider in ["openai", "tgi"]:
                        chunk = rechunk.openai_completion2ollama(chunk, stream, start_ts)

                    if hasattr(chunk, "model_dump_json"):
                        json_line = chunk.model_dump_json()
                    else:
                        json_line = json.dumps(chunk)
                    yield json_line.encode("utf-8") + b"\n"
            else:
                if provider == "huggingface":
                    response = rechunk.hf_generate2ollama(async_gen, stream, start_ts, model)
                    json_line = response.model_dump_json()
                elif provider in ["openai", "tgi"]:
                    response = rechunk.openai_completion2ollama(async_gen, stream, start_ts)
                    json_line = response.model_dump_json()
                else:  # ollama
                    json_line = async_gen.model_dump_json()

                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 5. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_generate_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 7. API route – Chat
# -------------------------------------------------------------
@app.post("/api/chat")
async def chat_proxy(request: Request):
    """
    Proxy a chat request to Ollama and stream the endpoint reply.
    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        messages = payload.get("messages")
        tools = payload.get("tools")
        stream = payload.get("stream")
        think = payload.get("think")
        _format = payload.get("format")
        keep_alive = payload.get("keep_alive")
        options = payload.get("options")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400, detail="Missing or invalid 'messages' field (must be a list)"
            )
        if options is not None and not isinstance(options, dict):
            raise HTTPException(
                status_code=400, detail="`options` must be a JSON object"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    provider = get_provider(endpoint)

    hf_client = None
    oclient = None
    client = None

    if provider == "huggingface":
        if not HF_AVAILABLE:
            raise HTTPException(status_code=500, detail="HuggingFace support not available. Install huggingface_hub.")

        # HuggingFace Inference API
        api_key = config.api_keys.get(endpoint, "")
        hf_client = InferenceClient(token=api_key)

        # Convert messages to prompt
        prompt = messages_to_prompt(messages)

        params = {
            "model": model,  # Format: "org/model-name"
            "prompt": prompt,
            "stream": stream if stream is not None else False,
        }

        # Optional parameters
        if options:
            if "temperature" in options:
                params["temperature"] = options["temperature"]
            if "num_predict" in options:
                params["max_new_tokens"] = options["num_predict"]
            if "top_p" in options:
                params["top_p"] = options["top_p"]

    elif provider in ["openai", "tgi"]:
        # OpenAI-compatible (includes TGI)
        if ":latest" in model:
            model = model.split(":latest")[0]

        params = {
            "messages": messages,
            "model": model,
        }
        optional_params = {
            "tools": tools,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream is not None else None,
            "max_tokens": options.get("num_predict") if options and "num_predict" in options else None,
            "frequency_penalty": options.get("frequency_penalty") if options and "frequency_penalty" in options else None,
            "presence_penalty": options.get("presence_penalty") if options and "presence_penalty" in options else None,
            "seed": options.get("seed") if options and "seed" in options else None,
            "stop": options.get("stop") if options and "stop" in options else None,
            "top_p": options.get("top_p") if options and "top_p" in options else None,
            "temperature": options.get("temperature") if options and "temperature" in options else None,
            "response_format": {"type": "json_schema", "json_schema": _format} if _format is not None else None
        }
        params.update({k: v for k, v in optional_params.items() if v is not None})
        oclient = openai.AsyncOpenAI(base_url=endpoint, default_headers=default_headers, api_key=config.api_keys[endpoint])

    else:  # ollama
        client = ollama.AsyncClient(host=endpoint)

    await increment_usage(endpoint, model)
    # 3. Async generator that streams chat data and decrements the counter
    async def stream_chat_response():
        try:
            # Initialize async generator based on provider
            if provider == "huggingface":
                start_ts = time.perf_counter()
                async_gen = hf_client.text_generation(**params, details=True)
            elif provider in ["openai", "tgi"]:
                start_ts = time.perf_counter()
                async_gen = await oclient.chat.completions.create(**params)
            else:  # ollama
                async_gen = await client.chat(model=model, messages=messages, tools=tools, stream=stream, think=think, format=_format, options=options, keep_alive=keep_alive)

            if stream:
                async for chunk in async_gen:
                    if provider == "huggingface":
                        chunk = rechunk.hf_chat2ollama(chunk, stream, start_ts, model)
                    elif provider in ["openai", "tgi"]:
                        chunk = rechunk.openai_chat_completion2ollama(chunk, stream, start_ts)

                    # Convert chunk to JSON
                    if hasattr(chunk, "model_dump_json"):
                        json_line = chunk.model_dump_json()
                    else:
                        json_line = json.dumps(chunk)
                    yield json_line.encode("utf-8") + b"\n"
            else:
                # Non-streaming response
                if provider == "huggingface":
                    response = rechunk.hf_chat2ollama(async_gen, stream, start_ts, model)
                    json_line = response.model_dump_json()
                elif provider in ["openai", "tgi"]:
                    response = rechunk.openai_chat_completion2ollama(async_gen, stream, start_ts)
                    json_line = response.model_dump_json()
                else:  # ollama
                    json_line = async_gen.model_dump_json()

                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    media_type = "application/x-ndjson" if stream else "application/json"
    return StreamingResponse(
        stream_chat_response(),
        media_type=media_type,
    )

# -------------------------------------------------------------
# 8. API route – Embedding - deprecated
# -------------------------------------------------------------
@app.post("/api/embeddings")
async def embedding_proxy(request: Request):
    """
    Proxy an embedding request to Ollama and reply with embeddings.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        prompt = payload.get("prompt")
        options = payload.get("options")
        keep_alive = payload.get("keep_alive")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Missing required field 'prompt'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    is_openai_endpoint = "/v1" in endpoint
    if is_openai_endpoint:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        client = openai.AsyncOpenAI(base_url=endpoint, api_key=config.api_keys[endpoint])
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)
    # 3. Async generator that streams embedding data and decrements the counter
    async def stream_embedding_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            if is_openai_endpoint:
                async_gen = await client.embeddings.create(input=prompt, model=model)
                async_gen = rechunk.openai_embeddings2ollama(async_gen)
            else:
                async_gen = await client.embeddings(model=model, prompt=prompt, options=options, keep_alive=keep_alive)
            if hasattr(async_gen, "model_dump_json"):
                json_line = async_gen.model_dump_json()
            else:
                json_line = json.dumps(async_gen)
            yield json_line.encode("utf-8") + b"\n"
        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 5. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_embedding_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 9. API route – Embed
# -------------------------------------------------------------
@app.post("/api/embed")
async def embed_proxy(request: Request):
    """
    Proxy an embed request to Ollama and reply with embeddings.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        _input = payload.get("input")
        truncate = payload.get("truncate")
        options = payload.get("options")
        keep_alive = payload.get("keep_alive")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not _input:
            raise HTTPException(
                status_code=400, detail="Missing required field 'input'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    provider = get_provider(endpoint)

    if provider == "huggingface":
        if not HF_AVAILABLE:
            raise HTTPException(status_code=500, detail="HuggingFace support not available. Install huggingface_hub.")

        api_key = config.api_keys.get(endpoint, "")
        hf_client = InferenceClient(token=api_key)
        oclient = None
        client = None

    elif provider in ["openai", "tgi"]:
        if ":latest" in model:
            model = model.split(":latest")[0]
        oclient = openai.AsyncOpenAI(base_url=endpoint, api_key=config.api_keys[endpoint])
        hf_client = None
        client = None

    else:  # ollama
        client = ollama.AsyncClient(host=endpoint)
        hf_client = None
        oclient = None

    await increment_usage(endpoint, model)

    # 3. Async generator that streams embed data and decrements the counter
    async def stream_embedding_response():
        try:
            if provider == "huggingface":
                # HF feature extraction
                embedding = await hf_client.feature_extraction(
                    text=_input,
                    model=model
                )
                async_gen = rechunk.hf_embeddings2ollama(embedding, model)

            elif provider in ["openai", "tgi"]:
                async_gen = await oclient.embeddings.create(input=_input, model=model)
                async_gen = rechunk.openai_embed2ollama(async_gen, model)

            else:  # ollama
                async_gen = await client.embed(model=model, input=_input, truncate=truncate, options=options, keep_alive=keep_alive)

            if hasattr(async_gen, "model_dump_json"):
                json_line = async_gen.model_dump_json()
            else:
                json_line = json.dumps(async_gen)
            yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_embedding_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 10. API route – Create
# -------------------------------------------------------------
@app.post("/api/create")
async def create_proxy(request: Request):
    """
    Proxy a create request to all Ollama endpoints and reply with deduplicated status.
    """
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        quantize = payload.get("quantize")
        from_ = payload.get("from")
        files = payload.get("files")
        adapters = payload.get("adapters")
        template = payload.get("template")
        license = payload.get("license")
        system = payload.get("system")
        parameters = payload.get("parameters")
        messages = payload.get("messages")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not from_ and not files:
            raise HTTPException(
                status_code=400, detail="You need to provide either from_ or files parameter!"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    
    status_lists = []
    for endpoint in config.endpoints:
        client = ollama.AsyncClient(host=endpoint)
        create = await client.create(model=model, quantize=quantize, from_=from_, files=files, adapters=adapters, template=template, license=license, system=system, parameters=parameters, messages=messages, stream=False)
        status_lists.append(create)

    combined_status = []
    for status_list in status_lists:
        combined_status += status_list

    final_status = list(dict.fromkeys(combined_status))

    return dict(final_status)

# -------------------------------------------------------------
# 11. API route – Show
# -------------------------------------------------------------
@app.post("/api/show")
async def show_proxy(request: Request, model: Optional[str] = None):
    """
    Proxy a model show request to Ollama and reply with ShowResponse.

    """
    try:
        body_bytes = await request.body()

        if not model:
            payload = json.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    #await increment_usage(endpoint, model)
    client = ollama.AsyncClient(host=endpoint)

    # 3. Proxy a simple show request
    show = await client.show(model=model)

    # 4. Return ShowResponse
    return show

# -------------------------------------------------------------
# 12. API route – Copy
# -------------------------------------------------------------
@app.post("/api/copy")
async def copy_proxy(request: Request, source: Optional[str] = None, destination: Optional[str] = None):
    """
    Proxy a model copy request to each Ollama endpoint and reply with Status Code.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()

        if not source and not destination:
            payload = json.loads(body_bytes.decode("utf-8"))
            src = payload.get("source")
            dst = payload.get("destination")
        else:
            src = source
            dst = destination
        
        if not src:
            raise HTTPException(
                status_code=400, detail="Missing required field 'source'"
            )
        if not dst:
            raise HTTPException(
                status_code=400, detail="Missing required field 'destination'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 3. Iterate over all endpoints to copy the model on each endpoint
    status_list = []
    for endpoint in config.endpoints:
        if "/v1" not in endpoint:
            client = ollama.AsyncClient(host=endpoint)
            # 4. Proxy a simple copy request
            copy = await client.copy(source=src, destination=dst)
            status_list.append(copy.status)

    # 4. Return with 200 OK if all went well, 404 if a single endpoint failed
    return Response(status_code=404 if 404 in status_list else 200)

# -------------------------------------------------------------
# 13. API route – Delete
# -------------------------------------------------------------
@app.delete("/api/delete")
async def delete_proxy(request: Request, model: Optional[str] = None):
    """
    Proxy a model delete request to each Ollama endpoint and reply with Status Code.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()

        if not model:
            payload = json.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Iterate over all endpoints to delete the model on each endpoint
    status_list = []
    for endpoint in config.endpoints:
        if "/v1" not in endpoint:
            client = ollama.AsyncClient(host=endpoint)
            # 3. Proxy a simple copy request
            copy = await client.delete(model=model)
            status_list.append(copy.status)
    
    # 4. Retrun 200 0K, if a single enpoint fails, respond with 404
    return Response(status_code=404 if 404 in status_list else 200)   

# -------------------------------------------------------------
# 14. API route – Pull
# -------------------------------------------------------------
@app.post("/api/pull")
async def pull_proxy(request: Request, model: Optional[str] = None):
    """
    Proxy a pull request to all Ollama endpoint and report status back.
    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()

        if not model:
            payload = json.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
            insecure = payload.get("insecure")
        else:
            insecure = None

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Iterate over all endpoints to pull the model
    status_list = []
    for endpoint in config.endpoints:
        if "/v1" not in endpoint:
            client = ollama.AsyncClient(host=endpoint)
            # 3. Proxy a simple pull request
            pull = await client.pull(model=model, insecure=insecure, stream=False)
            status_list.append(pull)

    combined_status = []
    for status in status_list:
        combined_status += status
    
    # 4. Report back a deduplicated status message
    final_status = list(dict.fromkeys(combined_status))

    return dict(final_status)

# -------------------------------------------------------------
# 14a. Helper function – Stream pull progress
# -------------------------------------------------------------
async def pull_model_stream(endpoint: str, model: str, insecure: bool, queue: asyncio.Queue):
    """
    Pull model auf einem Endpoint und sende Progress-Events zur Queue.
    """
    try:
        await queue.put(json.dumps({
            "type": "info",
            "endpoint": endpoint,
            "model": model,
            "status": "starting",
            "message": f"Starting pull on {endpoint}"
        }))

        client = ollama.AsyncClient(host=endpoint)
        async for progress in await client.pull(model=model, insecure=insecure, stream=True):
            await queue.put(json.dumps({
                "type": "progress",
                "endpoint": endpoint,
                "model": model,
                "completed": progress.get("completed"),
                "total": progress.get("total"),
                "digest": progress.get("digest"),
                "status": progress.get("status", "downloading")
            }))

        await queue.put(json.dumps({
            "type": "complete",
            "endpoint": endpoint,
            "model": model,
            "status": "success",
            "message": f"Successfully pulled {model}"
        }))
        return {"endpoint": endpoint, "status": "success"}

    except Exception as e:
        await queue.put(json.dumps({
            "type": "error",
            "endpoint": endpoint,
            "model": model,
            "status": "error",
            "message": str(e)
        }))
        return {"endpoint": endpoint, "status": "error", "detail": str(e)}

# -------------------------------------------------------------
# 14b. API route – Pull Stream (with endpoint selection)
# -------------------------------------------------------------
@app.post("/api/pull-stream")
async def pull_stream(request: Request):
    """
    Stream pull progress für ausgewählte Endpoints via SSE.
    """
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))
        model = payload.get("model")
        endpoints = payload.get("endpoints")
        insecure = payload.get("insecure", False)

        if not model:
            raise HTTPException(status_code=400, detail="Missing required field 'model'")

        # Default: alle Ollama-Endpoints (OpenAI ausschließen)
        if not endpoints:
            endpoints = [ep for ep in config.endpoints if "/v1" not in ep]
        else:
            endpoints = [ep for ep in endpoints if ep in config.endpoints and "/v1" not in ep]

        if not endpoints:
            raise HTTPException(status_code=400, detail="No valid Ollama endpoints selected")

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    async def event_generator():
        queue = asyncio.Queue(maxsize=100)

        try:
            tasks = [
                pull_model_stream(endpoint, model, insecure, queue)
                for endpoint in endpoints
            ]

            pending_count = len(tasks)
            gather_task = asyncio.create_task(
                asyncio.gather(*tasks, return_exceptions=True)
            )

            # Stream Events während Downloads laufen
            while pending_count > 0:
                if await request.is_disconnected():
                    gather_task.cancel()
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield f"data: {event}\n\n"

                    event_data = json.loads(event)
                    if event_data.get("type") in ["complete", "error"]:
                        pending_count -= 1

                except asyncio.TimeoutError:
                    if gather_task.done():
                        results = await gather_task
                        break
                    continue

            if not gather_task.done():
                results = await gather_task

            # Final Summary
            summary = {
                "type": "done",
                "model": model,
                "endpoints": endpoints,
                "results": [r for r in results if isinstance(r, dict)]
            }
            yield f"data: {json.dumps(summary)}\n\n"

        except Exception as e:
            error_event = {"type": "error", "message": f"Stream error: {str(e)}"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------------------
# 15. API route – Push
# -------------------------------------------------------------
@app.post("/api/push")
async def push_proxy(request: Request):
    """
    Proxy a push request to Ollama and respond the deduplicated Ollama endpoint replies.
    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        insecure = payload.get("insecure")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Iterate over all endpoints
    status_list = []
    for endpoint in config.endpoints:
        client = ollama.AsyncClient(host=endpoint)
        # 3. Proxy a simple push request
        push = await client.push(model=model, insecure=insecure, stream=False)
        status_list.append(push)

    combined_status = []
    for status in status_list:
        combined_status += status
    
    # 4. Report a deduplicated status
    final_status = list(dict.fromkeys(combined_status))

    return dict(final_status)


# -------------------------------------------------------------
# 16. API route – Version
# -------------------------------------------------------------
@app.get("/api/version")
async def version_proxy(request: Request):
    """
    Proxy a version request to Ollama and reply lowest version of all endpoints.

    """
    # 1. Query all endpoints for version
    tasks = [fetch.endpoint_details(ep, "/api/version", "version") for ep in config.endpoints if "/v1" not in ep]
    all_versions = await asyncio.gather(*tasks)

    def version_key(v):
        return tuple(map(int, v.split('.')))
    
    # 2. Return a JSONResponse with the min Version of all endpoints to maintain compatibility
    return JSONResponse(
        content={"version": str(min(all_versions, key=version_key))},
        status_code=200,
    )

# -------------------------------------------------------------
# 17. API route – tags
# -------------------------------------------------------------
@app.get("/api/tags")
async def tags_proxy(request: Request):
    """
    Proxy a tags request to Ollama endpoints and reply with a unique list of all models.

    """
    
    # 1. Query all endpoints for models
    tasks = [fetch.endpoint_details(ep, "/api/tags", "models") for ep in config.endpoints if "/v1" not in ep]
    tasks += [fetch.endpoint_details(ep, "/models", "data", config.api_keys[ep]) for ep in config.endpoints if "/v1" in ep]
    all_models = await asyncio.gather(*tasks)
    
    models = {'models': []}
    for modellist in all_models:
        for model in modellist:
            if not "model" in model.keys():  # Relable OpenAI models with Ollama Model.model from Model.id
                model['model'] = model['id'] + ":latest"
            else:
                model['id'] = model['model']
            if not "name" in model.keys():  # Relable OpenAI models with Ollama Model.name from Model.model to have model,name keys
                model['name'] = model['model']
            else:
                model['id'] = model['model']
        models['models'] += modellist
    
    # 2. Return a JSONResponse with a deduplicated list of unique models for inference
    return JSONResponse(
        content={"models": dedupe_on_keys(models['models'], ['digest','name','id'])},
        status_code=200,
    )

# -------------------------------------------------------------
# 18. API route – ps
# -------------------------------------------------------------
@app.get("/api/ps")
async def ps_proxy(request: Request):
    """
    Proxy a ps request to all Ollama endpoints and reply a unique list of all running models.

    """
    # 1. Query all endpoints for running models
    tasks = [fetch.endpoint_details(ep, "/api/ps", "models") for ep in config.endpoints if "/v1" not in ep]
    loaded_models = await asyncio.gather(*tasks)

    models = {'models': []}
    for modellist in loaded_models:
        models['models'] += modellist
    
    # 2. Return a JSONResponse with deduplicated currently deployed models
    return JSONResponse(
        content={"models": dedupe_on_keys(models['models'], ['digest'])},
        status_code=200,
    )

# -------------------------------------------------------------
# 19. Proxy usage route – for monitoring
# -------------------------------------------------------------
@app.get("/api/usage")
async def usage_proxy(request: Request):
    """
    Return a snapshot of the usage counter for each endpoint.
    Useful for debugging / monitoring.
    """
    return {"usage_counts": usage_counts}

# -------------------------------------------------------------
# 20. Proxy config route – for monitoring and frontent usage
# -------------------------------------------------------------
@app.get("/api/config")
async def config_proxy(request: Request):
    """
    Return a simple JSON object that contains the configured
    Ollama endpoints. The front‑end uses this to display
    which endpoints are being proxied.
    """
    async def check_endpoint(url: str):
        try:
            client: aiohttp.ClientSession = app_state["session"]
            if "/v1" in url:
                headers = {"Authorization": "Bearer " + config.api_keys[url]}
                async with client.get(f"{url}/models", headers=headers) as resp:
                    await _ensure_success(resp)
                    data = await resp.json()
            else:
                async with client.get(f"{url}/api/version") as resp:
                    await _ensure_success(resp)
                    data = await resp.json()
            if "/v1" in url:
                return {"url": url, "status": "ok", "version": "latest"}
            else:
                return {"url": url, "status": "ok", "version": data.get("version")}
        except Exception as e:
            return {"url": url, "status": "error", "detail": str(e)}

    results = await asyncio.gather(*[check_endpoint(ep) for ep in config.endpoints])
    return {"endpoints": results}

# -------------------------------------------------------------
# 21. API route – OpenAI compatible Embedding
# -------------------------------------------------------------
@app.post("/v1/embeddings")
async def openai_embedding_proxy(request: Request):
    """
    Proxy an OpenAI API compatible embedding request to Ollama and reply with embeddings.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        doc = payload.get("input")
        

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not doc:
            raise HTTPException(
                status_code=400, detail="Missing required field 'input'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    if "/v1" in endpoint:
        api_key = config.api_keys[endpoint]
    else:
        api_key = "ollama"
    base_url = ep2base(endpoint)
    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=api_key)

    # 3. Async generator that streams embedding data and decrements the counter
    async_gen = await oclient.embeddings.create(input=doc, model=model)
            
    await decrement_usage(endpoint, model)

    # 5. Return a StreamingResponse backed by the generator
    return async_gen

# -------------------------------------------------------------
# 22. API route – OpenAI compatible Chat Completions
# -------------------------------------------------------------
@app.post("/v1/chat/completions")
async def openai_chat_completions_proxy(request: Request):
    """
    Proxy an OpenAI API compatible chat completions request to Ollama and reply with a streaming response.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        messages = payload.get("messages")
        frequency_penalty = payload.get("frequency_penalty")
        presence_penalty = payload.get("presence_penalty")
        response_format = payload.get("response_format")
        seed = payload.get("seed")
        stop = payload.get("stop")
        stream = payload.get("stream")
        stream_options = payload.get("stream_options")
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        max_tokens = payload.get("max_tokens")
        max_completion_tokens = payload.get("max_completion_tokens")
        tools = payload.get("tools")

        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]

        params = {
            "messages": messages, 
            "model": model,
        }

        optional_params = {
            "tools": tools,
            "response_format": response_format,
            "stream_options": stream_options,
            "max_completion_tokens": max_completion_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop": stop,
            "stream": stream,
        }

        params.update({k: v for k, v in optional_params.items() if v is not None})
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400, detail="Missing required field 'messages' (must be a list)"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    base_url = ep2base(endpoint)
    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=config.api_keys[endpoint])
    
    # 3. Async generator that streams completions data and decrements the counter
    async def stream_ochat_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            async_gen = await oclient.chat.completions.create(**params)
            if stream == True:
                async for chunk in async_gen:
                    data = (
                        chunk.model_dump_json()
                        if hasattr(chunk, "model_dump_json")
                        else json.dumps(chunk)
                    )
                    if chunk.choices[0].delta.content is not None:
                        yield f"data: {data}\n\n".encode("utf-8")
                # Final DONE event
                #yield b"data: [DONE]\n\n"
            else:
                json_line = (
                    async_gen.model_dump_json()
                    if hasattr(async_gen, "model_dump_json")
                    else json.dumps(async_gen)
                )
                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_ochat_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 23. API route – OpenAI compatible Completions
# -------------------------------------------------------------
@app.post("/v1/completions")
async def openai_completions_proxy(request: Request):
    """
    Proxy an OpenAI API compatible chat completions request to Ollama and reply with a streaming response.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        prompt = payload.get("prompt")
        frequency_penalty = payload.get("frequency_penalty")
        presence_penalty = payload.get("presence_penalty")
        seed = payload.get("seed")
        stop = payload.get("stop")
        stream = payload.get("stream")
        stream_options = payload.get("stream_options")
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        max_tokens = payload.get("max_tokens")
        max_completion_tokens = payload.get("max_completion_tokens")
        suffix = payload.get("suffix")

        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]

        params = {
            "prompt": prompt, 
            "model": model,
        }

        optional_params = {
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "stream_options": stream_options,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "suffix": suffix
        }

        params.update({k: v for k, v in optional_params.items() if v is not None})

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Missing required field 'prompt'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    base_url = ep2base(endpoint)
    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=config.api_keys[endpoint])

    # 3. Async generator that streams completions data and decrements the counter
    async def stream_ocompletions_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            async_gen = await oclient.completions.create(**params)
            if stream == True:
                async for chunk in async_gen:
                    data = (
                        chunk.model_dump_json()
                        if hasattr(chunk, "model_dump_json")
                        else json.dumps(chunk)
                    )
                    yield f"data: {data}\n\n".encode("utf-8")
                # Final DONE event
                yield b"data: [DONE]\n\n"
            else:
                json_line = (
                    async_gen.model_dump_json()
                    if hasattr(async_gen, "model_dump_json")
                    else json.dumps(async_gen)
                )
                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_ocompletions_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 24. OpenAI API compatible models endpoint
# -------------------------------------------------------------
@app.get("/v1/models")
async def openai_models_proxy(request: Request):
    """
    Proxy an OpenAI API models request to Ollama endpoints and reply with a unique list of all models.

    """
    # 1. Query all endpoints for models
    tasks = [fetch.endpoint_details(ep, "/api/tags", "models") for ep in config.endpoints if "/v1" not in ep]
    tasks += [fetch.endpoint_details(ep, "/models", "data", config.api_keys[ep]) for ep in config.endpoints if "/v1" in ep]
    all_models = await asyncio.gather(*tasks)
    
    models = {'data': []}
    for modellist in all_models:
        for model in modellist:
            if not "id" in model.keys():  # Relable Ollama models with OpenAI Model.id from Model.name
                model['id'] = model['name']
            else:
                model['name'] = model['id']
        models['data'] += modellist
    
    # 2. Return a JSONResponse with a deduplicated list of unique models for inference
    return JSONResponse(
        content={"data": dedupe_on_keys(models['data'], ['name'])},
        status_code=200,
    )

# -------------------------------------------------------------
# 25. Serve the static front‑end
# -------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def redirect_favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the dynamic NOMYO Router dashboard listing the configured endpoints
    and the models details, availability & task status.
    """
    return HTMLResponse(content=open("static/index.html", "r").read(), status_code=200)

# -------------------------------------------------------------
# 26. Healthendpoint
# -------------------------------------------------------------
@app.get("/health")
async def health_proxy(request: Request):
    """
    Health‑check endpoint for monitoring the proxy.

    * Queries each configured endpoint for its `/api/version` response.
    * Returns a JSON object containing:
        - `status`: "ok" if every endpoint replied, otherwise "error".
        - `endpoints`: a mapping of endpoint URL → `{status, version|detail}`.
    * The HTTP status code is 200 when everything is healthy, 503 otherwise.
    """
    # Run all health checks in parallel
    tasks = [fetch.endpoint_details(ep, "/api/version", "version") for ep in config.endpoints]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    health_summary = {}
    overall_ok = True

    for ep, result in zip(config.endpoints, results):
        if isinstance(result, Exception):
            # Endpoint did not respond / returned an error
            health_summary[ep] = {"status": "error", "detail": str(result)}
            overall_ok = False
        else:
            # Successful response – report the reported version
            health_summary[ep] = {"status": "ok", "version": result}

    response_payload = {
        "status": "ok" if overall_ok else "error",
        "endpoints": health_summary,
    }

    http_status = 200 if overall_ok else 503
    return JSONResponse(content=response_payload, status_code=http_status)

# -------------------------------------------------------------
# 27. SSE route for usage broadcasts
# -------------------------------------------------------------
@app.get("/api/usage-stream")
async def usage_stream(request: Request):
    """
    Server‑Sent‑Events that emits a JSON payload every time the
    global `usage_counts` dictionary changes.
    """
    async def event_generator():
        # The queue that receives *every* new snapshot
        queue = await subscribe()
        try:
            while True:
                # If the client disconnects, cancel the loop
                if await request.is_disconnected():
                    break
                data = await queue.get()
                if data is None:
                    break
                # Send the data as a single SSE message
                yield f"data: {data}\n\n"
        finally:
            # Clean‑up: unsubscribe from the broadcast channel
            await unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------------------
# 28. Hardware & Model Search Endpoints
# -------------------------------------------------------------
@app.get("/api/hardware")
async def get_hardware():
    """Get current hardware stats."""
    stats = await get_hardware_stats()
    return {
        "ram": {
            "total_gb": stats.ram_total_gb,
            "available_gb": stats.ram_available_gb,
            "used_percent": stats.ram_used_percent
        },
        "disk": {
            "total_gb": stats.disk_total_gb,
            "available_gb": stats.disk_available_gb,
            "used_percent": stats.disk_used_percent
        },
        "cpu": {
            "cores": stats.cpu_cores,
            "gpu": stats.gpu_available
        },
        "platform": stats.platform
    }

@app.post("/api/check-compatibility")
async def check_compatibility(request: Request):
    """Check if a model is compatible with current hardware."""
    body = await request.json()
    model_name = body.get("model_name")
    provider = body.get("provider", "ollama")

    if not model_name:
        raise HTTPException(status_code=400, detail="model_name required")

    result = await check_model_compatibility(model_name, provider)
    return result

@app.get("/api/search/ollama")
async def search_ollama_models(query: str = "", limit: int = 20):
    """Search curated Ollama models."""
    import json
    from pathlib import Path

    # Load curated models list
    models_file = Path(__file__).parent / "data" / "ollama_models.json"

    if not models_file.exists():
        return {"models": [], "error": "Models list not found"}

    with open(models_file, "r") as f:
        data = json.load(f)
        all_models = data.get("models", [])

    # Filter by query
    query_lower = query.lower()
    if query:
        filtered = [
            m for m in all_models
            if query_lower in m["name"].lower() or
               query_lower in m["description"].lower() or
               any(query_lower in tag for tag in m.get("tags", []))
        ]
    else:
        filtered = all_models[:limit]

    # Add compatibility info
    for model in filtered[:limit]:
        compat = await check_model_compatibility(model["name"], "ollama")
        model["compatibility"] = {
            "status": compat["status"],
            "compatible": compat["compatible"],
            "message": compat["checks"].get("disk_space", {}).get("message", "")
        }

    return {"models": filtered[:limit], "total": len(filtered)}

@app.get("/api/search/hf")
async def search_hf_models(query: str = "", task: str = "text-generation", limit: int = 20):
    """Search HuggingFace models."""
    if not HF_AVAILABLE or HfApi is None:
        return {"models": [], "error": "HuggingFace integration not available"}

    try:
        api = HfApi()

        # Search models
        models = api.list_models(
            search=query if query else None,
            task=task if task else None,
            limit=limit,
            sort="downloads",
            direction=-1
        )

        # Filter for inference API compatible
        results = []
        for model in models:
            if hasattr(model, 'pipeline_tag') and model.pipeline_tag:
                # Safely get description from card_data
                description = ""
                if hasattr(model, 'card_data') and model.card_data:
                    if isinstance(model.card_data, dict):
                        description = model.card_data.get('description', '')

                results.append({
                    "id": model.id,
                    "author": model.author or "unknown",
                    "downloads": getattr(model, 'downloads', 0),
                    "likes": getattr(model, 'likes', 0),
                    "pipeline_tag": model.pipeline_tag,
                    "inference_api": True,
                    "description": description
                })

        return {"models": results[:limit], "total": len(results)}

    except Exception as e:
        return {"models": [], "error": str(e)}

# -------------------------------------------------------------
# 29. FastAPI startup/shutdown events
# -------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    global config
    # Load YAML config (or use defaults if not present)
    config = Config.from_yaml(Path("config.yaml"))
    print(f"Loaded configuration:\n endpoints={config.endpoints},\n "
          f"max_concurrent_connections={config.max_concurrent_connections}")
    
    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=512, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=5, connect=5, sock_read=120, sock_connect=5)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    app_state["connector"] = connector
    app_state["session"] = session


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await close_all_sse_queues()
    await app_state["session"].close()