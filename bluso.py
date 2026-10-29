#!/usr/bin/env python3
"""Bluso: a dependency-free, local-first LLM observability console."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
STATE_DIR = ROOT / ".bluso"
STATE_DIR.mkdir(exist_ok=True)

DEFAULT_LLAMA_PORTS = (8081, 9931, 8080)
DEFAULT_OPENAI_PORTS = (1234, 8000, 5000, 5001, 4891, 1337)
MODEL_SUFFIXES = {".gguf"}
DISCOVERY_TTL = 12.0
GGUF_STRING = 8
GGUF_ARRAY = 9


def now_ms() -> int:
    return int(time.time() * 1000)


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def http_json(url: str, method: str = "GET", payload: Any = None, timeout: float = 1.2) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "bluso/0.1"}
    body = None
    if payload is not None:
        body = json_bytes(payload)
        headers["Content-Type"] = "application/json"
    req = Request(url, data=body, method=method, headers=headers)
    with urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def safe_get(url: str, timeout: float = 0.8) -> Optional[Any]:
    try:
        return http_json(url, timeout=timeout)
    except (OSError, HTTPError, URLError, ValueError, TimeoutError):
        return None


def port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def executable_candidates(names: Iterable[str]) -> List[str]:
    home = Path.home()
    paths: List[str] = []
    for name in names:
        found = shutil.which(name)
        if found:
            paths.append(found)
        for candidate in (home / ".local/bin" / name, Path("/opt/homebrew/bin") / name,
                          Path("/usr/local/bin") / name, Path("/usr/bin") / name):
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                paths.append(str(candidate))
    return list(dict.fromkeys(paths))


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


class GGUFReader:
    """Reads GGUF metadata only; tensor payloads are never loaded into memory."""

    scalar_formats = {
        0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
        6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
    }

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def _read(self, amount: int) -> bytes:
        data = self.handle.read(amount)
        if len(data) != amount:
            raise ValueError("Unexpected end of GGUF metadata")
        return data

    def _u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def _string(self) -> str:
        length = self._u64()
        if length > 64 * 1024 * 1024:
            raise ValueError("Implausible GGUF string length")
        return self._read(length).decode("utf-8", errors="replace")

    def _value(self, kind: int, depth: int = 0) -> Any:
        if depth > 3:
            raise ValueError("GGUF metadata nesting is too deep")
        if kind == GGUF_STRING:
            return self._string()
        if kind == GGUF_ARRAY:
            item_kind = struct.unpack("<I", self._read(4))[0]
            count = self._u64()
            if count > 2_000_000:
                raise ValueError("Implausible GGUF array length")
            # Large tokenizer arrays are real but not useful to the dashboard.
            if count > 4096 and item_kind == GGUF_STRING:
                for _ in range(count):
                    length = self._u64()
                    self.handle.seek(length, 1)
                return {"count": count, "omitted": True}
            values = [self._value(item_kind, depth + 1) for _ in range(count)]
            return values
        fmt = self.scalar_formats.get(kind)
        if not fmt:
            raise ValueError(f"Unsupported GGUF value type {kind}")
        return struct.unpack("<" + fmt, self._read(struct.calcsize(fmt)))[0]

    def read(self) -> Dict[str, Any]:
        with self.path.open("rb") as handle:
            self.handle = handle
            if self._read(4) != b"GGUF":
                raise ValueError("Not a GGUF file")
            version = struct.unpack("<I", self._read(4))[0]
            tensor_count = self._u64()
            metadata_count = self._u64()
            if metadata_count > 100_000:
                raise ValueError("Implausible GGUF metadata count")
            metadata: Dict[str, Any] = {}
            for _ in range(metadata_count):
                key = self._string()
                kind = struct.unpack("<I", self._read(4))[0]
                value = self._value(kind)
                if not key.startswith("tokenizer.ggml.tokens") and not key.startswith("tokenizer.ggml.merges"):
                    metadata[key] = value
            return {"gguf_version": version, "tensor_count": tensor_count, "metadata": metadata}


def summarize_gguf(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    result: Dict[str, Any] = {
        "id": "gguf:" + str(path), "name": path.stem, "path": str(path),
        "format": "GGUF", "bytes": stat.st_size, "size": human_bytes(stat.st_size),
        "modified": int(stat.st_mtime),
    }
    try:
        parsed = GGUFReader(path).read()
        meta = parsed["metadata"]
        arch = str(meta.get("general.architecture", "unknown"))
        prefix = arch + "."
        parameter_count = meta.get("general.parameter_count")
        if parameter_count is None:
            size_label = str(meta.get("general.size_label", ""))
            size_match = re.fullmatch(r"([\d.]+)\s*([BM])", size_label, re.IGNORECASE)
            if size_match:
                parameter_count = int(float(size_match.group(1)) * (1_000_000_000 if size_match.group(2).upper() == "B" else 1_000_000))
        compact_metadata = {}
        for key, value in meta.items():
            if isinstance(value, (list, dict)):
                continue
            if isinstance(value, str) and len(value) > 2000:
                value = f"<omitted {len(value):,}-character value>"
            compact_metadata[key] = value
        result.update({
            "display_name": meta.get("general.name", path.stem),
            "architecture": arch,
            "parameters": parameter_count,
            "quantization": meta.get("general.file_type"),
            "layers": meta.get(prefix + "block_count"),
            "context_train": meta.get(prefix + "context_length"),
            "embedding": meta.get(prefix + "embedding_length"),
            "attention_heads": meta.get(prefix + "attention.head_count"),
            "attention_kv_heads": meta.get(prefix + "attention.head_count_kv"),
            "feed_forward": meta.get(prefix + "feed_forward_length"),
            "rope_dimensions": meta.get(prefix + "rope.dimension_count"),
            "vocab_size": meta.get("tokenizer.ggml.tokens", {}).get("count") if isinstance(meta.get("tokenizer.ggml.tokens"), dict) else None,
            "tensor_count": parsed["tensor_count"], "gguf_version": parsed["gguf_version"],
            "metadata": compact_metadata,
        })
    except Exception as exc:
        result["metadata_error"] = str(exc)
    return result


@dataclass
class Runtime:
    id: str
    kind: str
    name: str
    base_url: str
    online: bool
    version: Optional[str] = None
    models: Optional[List[Dict[str, Any]]] = None
    features: Optional[List[str]] = None


class Discovery:
    def __init__(self):
        self.lock = threading.Lock()
        self.updated = 0.0
        self.cached: Dict[str, Any] = {}

    def model_paths(self) -> List[Path]:
        home = Path.home()
        roots = [
            home / "Downloads/AIlocalMLengineering",
            home / "Downloads", home / "Models", home / "models",
            home / "Library/Application Support/LM Studio/models",
            home / ".cache/lm-studio/models",
        ]
        seen: set = set()
        results: List[Path] = []
        for root in roots:
            if not root.exists():
                continue
            try:
                for base, dirs, files in os.walk(str(root)):
                    depth = len(Path(base).relative_to(root).parts)
                    if depth >= 3:
                        dirs[:] = []
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for filename in files:
                        path = Path(base) / filename
                        if path.suffix.lower() in MODEL_SUFFIXES and path not in seen:
                            seen.add(path)
                            results.append(path)
            except OSError:
                continue
        return sorted(results, key=lambda p: ("saluki" not in p.name.lower(), p.name.lower()))[:40]

    def probe_runtime(self, kind: str, base: str) -> Optional[Runtime]:
        if kind == "ollama":
            version_data = safe_get(base + "/api/version")
            tags = safe_get(base + "/api/tags") if version_data else None
            if version_data is not None:
                models = []
                for item in (tags or {}).get("models", []):
                    models.append({"name": item.get("name"), "size": item.get("size"), "details": item.get("details", {})})
                return Runtime("ollama:" + base, kind, "Ollama", base, True,
                               str(version_data.get("version", "unknown")), models,
                               ["streaming", "chat", "model-list"])
            return None
        health = safe_get(base + "/health")
        models_data = safe_get(base + "/v1/models")
        if health is not None or models_data is not None:
            props = safe_get(base + "/props")
            is_llama = props is not None
            model_items = (models_data or {}).get("data", []) if isinstance(models_data, dict) else []
            models = [{"name": m.get("id"), "meta": m.get("meta", {})} for m in model_items]
            if props and props.get("model_path"):
                models.insert(0, {"name": Path(props["model_path"]).name, "path": props["model_path"]})
            return Runtime(("llamacpp:" if is_llama else "openai:") + base,
                           "llamacpp" if is_llama else "openai",
                           "llama.cpp" if is_llama else "OpenAI-compatible", base, True,
                           None, models, ["streaming", "chat", "logprobs"] if is_llama else ["streaming", "chat"])
        return None

    def discover(self, force: bool = False) -> Dict[str, Any]:
        with self.lock:
            if not force and self.cached and time.time() - self.updated < DISCOVERY_TTL:
                return self.cached
            runtimes: List[Runtime] = []
            targets = [("auto", f"http://127.0.0.1:{port}") for port in dict.fromkeys(DEFAULT_LLAMA_PORTS + DEFAULT_OPENAI_PORTS)]
            targets.append(("ollama", "http://127.0.0.1:11434"))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as executor:
                futures = [executor.submit(self.probe_runtime, kind, base) for kind, base in targets]
                for future in futures:
                    runtime = future.result()
                    if runtime and all(r.base_url != runtime.base_url for r in runtimes):
                        runtimes.append(runtime)
            models = [summarize_gguf(path) for path in self.model_paths()]
            binaries = {
                "llamacpp": executable_candidates(("llama", "llama-server")),
                "ollama": executable_candidates(("ollama",)),
            }
            self.cached = {
                "runtimes": [asdict(item) for item in runtimes], "models": models,
                "binaries": binaries, "updated_at": now_ms(),
            }
            self.updated = time.time()
            return self.cached


DISCOVERY = Discovery()
RUNTIME_PROCESS: Optional[subprocess.Popen] = None
RUNTIME_LOCK = threading.Lock()
BOOT_LOG = STATE_DIR / "llama.log"


def default_saluki(discovery: Optional[Dict[str, Any]] = None) -> Optional[str]:
    data = discovery or DISCOVERY.discover()
    models = data.get("models", [])
    preferred = next((m for m in models if "saluki" in m.get("name", "").lower()), None)
    return preferred.get("path") if preferred else (models[0].get("path") if models else None)


def start_llamacpp(model_path: Optional[str] = None, port: int = 8081) -> Dict[str, Any]:
    global RUNTIME_PROCESS
    with RUNTIME_LOCK:
        if port_open("127.0.0.1", port):
            return {"ok": True, "state": "already-running", "url": f"http://127.0.0.1:{port}"}
        discovery = DISCOVERY.discover(force=True)
        path = model_path or default_saluki(discovery)
        binaries = discovery.get("binaries", {}).get("llamacpp", [])
        if not path or not Path(path).is_file():
            return {"ok": False, "error": "No GGUF model was found"}
        if not binaries:
            return {"ok": False, "error": "llama.cpp was not found"}
        binary = binaries[0]
        try:
            context_size = max(512, min(int(os.environ.get("BLUSO_CONTEXT", "4096")), 262144))
        except ValueError:
            context_size = 4096
        command = [binary]
        if Path(binary).name == "llama":
            command.append("serve")
        command += [
            "-m", path, "--host", "127.0.0.1", "--port", str(port),
            "--metrics", "--slots", "--no-webui", "--cors-origins", "localhost", "--jinja", "-ngl", "all",
            "-fa", "on", "-c", str(context_size), "-t", "4", "-tb", "4", "-np", "1",
            "-ctk", "q8_0", "-ctv", "q8_0", "--temp", "1.0", "--top-p", "0.95",
            "--top-k", "20", "--presence-penalty", "1.1",
        ]
        try:
            log = BOOT_LOG.open("ab", buffering=0)
            RUNTIME_PROCESS = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, cwd=str(Path(path).parent))
            DISCOVERY.updated = 0
            return {"ok": True, "state": "starting", "pid": RUNTIME_PROCESS.pid,
                    "url": f"http://127.0.0.1:{port}", "model": path, "context_size": context_size,
                    "log": str(BOOT_LOG)}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}


class SystemSampler:
    def __init__(self):
        self.last_cpu: Optional[Tuple[int, int]] = None

    def _linux(self) -> Dict[str, Any]:
        cpu_percent = None
        try:
            fields = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
            idle, total = fields[3] + fields[4], sum(fields)
            if self.last_cpu:
                idle_delta, total_delta = idle - self.last_cpu[0], total - self.last_cpu[1]
                cpu_percent = 100.0 * (1 - idle_delta / max(1, total_delta))
            self.last_cpu = (idle, total)
        except (OSError, ValueError, IndexError):
            pass
        memory: Dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                memory[key] = int(value.strip().split()[0]) * 1024
        except (OSError, ValueError):
            pass
        total = memory.get("MemTotal", 0)
        available = memory.get("MemAvailable", 0)
        return {"cpu_percent": cpu_percent, "memory_total": total, "memory_used": total - available,
                "memory_percent": 100 * (total - available) / total if total else None}

    def _mac(self) -> Dict[str, Any]:
        cpu_percent = None
        memory_total = 0
        memory_used = 0
        try:
            output = subprocess.check_output(["/usr/bin/top", "-l", "1", "-n", "0"], text=True, timeout=2)
            match = re.search(r"CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys", output)
            if match:
                cpu_percent = float(match.group(1)) + float(match.group(2))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        try:
            memory_total = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True).strip())
            page_size = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.pagesize"], text=True).strip())
            vm = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
            pages: Dict[str, int] = {}
            for line in vm.splitlines()[1:]:
                match = re.match(r"([^:]+):\s+(\d+)", line)
                if match:
                    pages[match.group(1)] = int(match.group(2))
            # Approximate Activity Monitor's "memory used" rather than treating
            # macOS's reclaimable file cache as consumed RAM.
            active = pages.get("Pages active", 0)
            wired = pages.get("Pages wired down", 0)
            compressed = pages.get("Pages occupied by compressor", 0)
            memory_used = min(memory_total, max(0, (active + wired + compressed) * page_size))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return {"cpu_percent": cpu_percent, "memory_total": memory_total, "memory_used": memory_used,
                "memory_percent": 100 * memory_used / memory_total if memory_total else None,
                "unified_memory": True}

    def process_stats(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            output = subprocess.check_output(["ps", "-axo", "pid=,pcpu=,pmem=,rss=,comm="], text=True, timeout=2)
            for line in output.splitlines():
                parts = line.strip().split(None, 4)
                if len(parts) != 5:
                    continue
                pid, cpu, mem, rss, command = parts
                label = Path(command).name.lower()
                if "llama" in label or "ollama" in label:
                    rows.append({"pid": int(pid), "cpu_percent": float(cpu), "memory_percent": float(mem),
                                 "rss": int(rss) * 1024, "command": command})
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return rows

    def sample(self) -> Dict[str, Any]:
        data = self._mac() if sys.platform == "darwin" else self._linux()
        load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
        data.update({"load": list(load), "platform": platform.system(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count(), "processes": self.process_stats(), "sampled_at": now_ms()})
        return data


SYSTEM = SystemSampler()


def parse_prometheus(text: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([-+\deE.]+)$", line)
        if match:
            try:
                result[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    return result


def fetch_text(url: str, timeout: float = 1.0) -> Optional[str]:
    try:
        with urlopen(Request(url, headers={"User-Agent": "bluso/0.1"}), timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, HTTPError, URLError, TimeoutError):
        return None


def telemetry() -> Dict[str, Any]:
    discovery = DISCOVERY.discover()
    runtime_details = []
    for runtime in discovery.get("runtimes", []):
        item = dict(runtime)
        base = runtime["base_url"]
        if runtime["kind"] == "llamacpp":
            item["slots"] = safe_get(base + "/slots")
            raw_metrics = fetch_text(base + "/metrics")
            item["metrics"] = parse_prometheus(raw_metrics) if raw_metrics else {}
            props = safe_get(base + "/props") or {}
            item["props"] = {key: props.get(key) for key in
                             ("model_alias", "model_ftype", "model_path", "modalities", "build_info", "is_sleeping")}
            item["props"]["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        elif runtime["kind"] == "ollama":
            item["running"] = safe_get(base + "/api/ps")
        runtime_details.append(item)
    return {"system": SYSTEM.sample(), "runtimes": runtime_details,
            "models": discovery.get("models", []), "sampled_at": now_ms()}


def probability_value(item: Dict[str, Any]) -> float:
    if "prob" in item:
        return float(item.get("prob") or 0)
    value = float(item.get("logprob") or -100)
    return math.exp(max(-100, min(0, value)))


def normalize_logprobs(logprobs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(logprobs, dict):
        return None
    entries = logprobs.get("content")
    if not isinstance(entries, list) or not entries:
        return None
    selected = entries[-1]
    top = selected.get("top_probs") or selected.get("top_logprobs") or []
    alternatives = [{"token": p.get("token", ""), "id": p.get("id"),
                     "probability": probability_value(p), "logprob": p.get("logprob")}
                    for p in top[:10]]
    return {"token": selected.get("token", ""), "id": selected.get("id"),
            "probability": probability_value(selected), "alternatives": alternatives}


def iter_sse(response) -> Iterable[Dict[str, Any]]:
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if not value or value == "[DONE]":
            continue
        try:
            yield json.loads(value)
        except ValueError:
            continue


class BlusoHandler(BaseHTTPRequestHandler):
    server_version = "Bluso/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, value: Any, status: int = 200) -> None:
        body = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = min(int(self.headers.get("Content-Length", "0")), 2 * 1024 * 1024)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / relative).resolve()
        if STATIC.resolve() not in target.parents and target != STATIC.resolve():
            self.send_error(403)
            return
        if not target.is_file():
            target = STATIC / "index.html"
        body = target.read_bytes()
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/discover":
            self._json(DISCOVERY.discover(force=True))
        elif path == "/api/telemetry":
            self._json(telemetry())
        elif path == "/api/health":
            self._json({"ok": True, "name": "Bluso", "version": "0.1.0", "time": now_ms()})
        elif path == "/api/logs":
            try:
                lines = BOOT_LOG.read_text(errors="replace").splitlines()[-160:]
            except OSError:
                lines = []
            self._json({"lines": lines})
        else:
            self._static(path)

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / relative).resolve()
        if (STATIC.resolve() not in target.parents and target != STATIC.resolve()) or not target.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
        except (ValueError, UnicodeDecodeError):
            self._json({"error": "Invalid JSON"}, 400)
            return
        if path == "/api/runtime/start":
            result = start_llamacpp(body.get("model_path"), int(body.get("port", 8081)))
            self._json(result, 200 if result.get("ok") else 400)
        elif path == "/api/generate":
            self._generate(body)
        else:
            self._json({"error": "Not found"}, 404)

    def _event(self, value: Any) -> None:
        self.wfile.write(json_bytes(value) + b"\n")
        self.wfile.flush()

    def _generate(self, body: Dict[str, Any]) -> None:
        discovery = DISCOVERY.discover(force=True)
        runtimes = discovery.get("runtimes", [])
        requested = body.get("runtime_id")
        runtime = next((r for r in runtimes if r.get("id") == requested), None)
        if not runtime:
            runtime = next((r for r in runtimes if r.get("kind") == "llamacpp"), None)
        if not runtime:
            self._json({"error": "No online model runtime. Start llama.cpp first."}, 503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        generation_id = "run_" + uuid.uuid4().hex[:12]
        self._event({"type": "start", "id": generation_id, "runtime": runtime, "at": now_ms()})
        try:
            if runtime["kind"] == "ollama":
                self._generate_ollama(runtime, body)
            else:
                self._generate_openai(runtime, body)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._event({"type": "error", "message": str(exc)})

    def _generate_openai(self, runtime: Dict[str, Any], body: Dict[str, Any]) -> None:
        prompt = str(body.get("prompt", "")).strip()
        system = str(body.get("system", "You are a precise, helpful assistant."))
        payload = {
            "model": body.get("model", "local-model"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max(1, min(int(body.get("max_tokens", 256)), 4096)),
            "temperature": max(0.0, min(float(body.get("temperature", 0.8)), 2.0)),
            "top_p": max(0.0, min(float(body.get("top_p", 0.95)), 1.0)),
        }
        if runtime["kind"] == "llamacpp":
            payload.update({"n_probs": 10, "top_logprobs": 10, "logprobs": True,
                            "post_sampling_probs": False, "timings_per_token": True})
        req = Request(runtime["base_url"] + "/v1/chat/completions", data=json_bytes(payload), method="POST",
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        started = time.perf_counter()
        token_count = 0
        latest_timings: Dict[str, Any] = {}
        with urlopen(req, timeout=600) as response:
            for chunk in iter_sse(response):
                if chunk.get("timings"):
                    latest_timings = chunk["timings"]
                choices = chunk.get("choices") or []
                if not choices:
                    if chunk.get("usage"):
                        self._event({"type": "usage", "usage": chunk["usage"]})
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                probs = normalize_logprobs(choice.get("logprobs"))
                if text or reasoning or probs:
                    token_count += 1 if probs or text else 0
                    elapsed = max(0.001, time.perf_counter() - started)
                    speed = latest_timings.get("predicted_per_second") or token_count / elapsed
                    self._event({"type": "token", "text": text, "reasoning": reasoning,
                                 "probabilities": probs, "index": token_count,
                                 "elapsed_ms": int(elapsed * 1000), "tokens_per_second": speed,
                                 "timings": latest_timings})
                if choice.get("finish_reason"):
                    self._event({"type": "finish", "reason": choice["finish_reason"],
                                 "tokens": token_count, "elapsed_ms": int((time.perf_counter() - started) * 1000),
                                 "timings": latest_timings})

    def _generate_ollama(self, runtime: Dict[str, Any], body: Dict[str, Any]) -> None:
        model = body.get("model")
        if not model:
            models = runtime.get("models") or []
            model = models[0].get("name") if models else None
        if not model:
            raise ValueError("Ollama has no installed model")
        payload = {"model": model, "stream": True,
                   "messages": [{"role": "system", "content": body.get("system", "You are a helpful assistant.")},
                                {"role": "user", "content": body.get("prompt", "")}],
                   "options": {"temperature": body.get("temperature", 0.8), "top_p": body.get("top_p", 0.95),
                               "num_predict": body.get("max_tokens", 256)}}
        req = Request(runtime["base_url"] + "/api/chat", data=json_bytes(payload), method="POST",
                      headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        token_count = 0
        with urlopen(req, timeout=600) as response:
            for raw in response:
                try:
                    chunk = json.loads(raw)
                except ValueError:
                    continue
                text = (chunk.get("message") or {}).get("content", "")
                reasoning = (chunk.get("message") or {}).get("thinking", "")
                if text or reasoning:
                    token_count += 1
                    elapsed = max(0.001, time.perf_counter() - started)
                    self._event({"type": "token", "text": text, "reasoning": reasoning,
                                 "probabilities": None, "index": token_count,
                                 "elapsed_ms": int(elapsed * 1000), "tokens_per_second": token_count / elapsed,
                                 "note": "Ollama does not expose token logprobs through its public API."})
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", token_count)
                    eval_duration = chunk.get("eval_duration", 0)
                    self._event({"type": "finish", "reason": chunk.get("done_reason", "stop"),
                                 "tokens": eval_count, "elapsed_ms": int((time.perf_counter() - started) * 1000),
                                 "timings": {"predicted_n": eval_count,
                                             "predicted_per_second": eval_count / (eval_duration / 1e9) if eval_duration else None}})


def open_browser(url: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", url])
    elif sys.platform.startswith("linux") and shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", url])


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-first LLM observability console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3417)
    parser.add_argument("--auto-start", action="store_true", help="Start the discovered Saluki GGUF with llama.cpp")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.auto_start:
        result = start_llamacpp()
        print("llama.cpp:", result.get("state", result.get("error")))
    server = ThreadingHTTPServer((args.host, args.port), BlusoHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Bluso is live at {url}")
    print("  Press Ctrl+C to stop the dashboard. The model runtime is left running.\n")
    if not args.no_browser:
        threading.Timer(0.8, open_browser, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping Bluso…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
