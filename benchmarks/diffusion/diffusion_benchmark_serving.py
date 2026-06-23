# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark online serving for diffusion models (Image/Video Generation).
If you want to use i2v, i2i dataset, you should `uv pip install gdown` first

Supports multiple backends:
    - vllm-omni: Uses /v1/chat/completions endpoint (default)
    - openai: Uses /v1/images/generations endpoint
    - v1/videos: Use /v1/videos endpoint

Usage:
    # Video (vllm-omni backend)
    t2v:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend vllm-omni --dataset vbench --task t2v --num-prompts 10 \
        --height 480 --width 640 --fps 16 --num-frames 80

    i2v:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend vllm-omni --dataset vbench --task i2v --num-prompts 10


    # Image (vllm-omni backend)
    t2i:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend vllm-omni --dataset vbench --task t2i --num-prompts 10 \
        --height 1024 --width 1024

    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend vllm-omni --dataset random --task t2i --num-prompts 1 \
        --max-concurrency 1 --enable-negative-prompt \
        --random-request-config '[
            {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
            {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
            {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
            {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
        ]'

    i2i:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend vllm-omni --dataset vbench --task i2i --num-prompts 10

    # Image (openai backend)
    t2i:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend openai --dataset vbench --task t2i --num-prompts 10 \
        --height 1024 --width 1024 --port 3000

    # Video (v1/vedeos)
    t2v:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend v1/videos --dataset random --task t2v --num-prompts 1 \
        --max-concurrency 1 --enable-negative-prompt \
        --random-request-config '[
            {"width":854,"height":480,"num_inference_steps":18,"num_frames":120,"fps":24,"weight":1}
        ]'


"""

import argparse
import ast
import asyncio
import glob
import json
import logging
import os
import random
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import Any

import aiohttp
import numpy as np
import requests
from backends import RequestFuncInput, RequestFuncOutput, backends_function_mapping
from PIL import Image
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    def __init__(self, args, api_url: str, model: str):
        self.args = args
        self.api_url = api_url
        self.model = model

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> RequestFuncInput:
        pass

    @abstractmethod
    def get_requests(self) -> list[RequestFuncInput]:
        pass


class VBenchDataset(BaseDataset):
    """
    Dataset loader for VBench prompts.
    Supports t2v, i2v.
    """

    T2V_PROMPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/prompts_per_dimension/subject_consistency.txt"
    )
    I2V_DOWNLOAD_SCRIPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/download_data.sh"
    )

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm-omni")
        self.items = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if self.args.task == "t2v":
            return self._load_t2v_prompts()
        elif self.args.task in ["i2v", "ti2v", "ti2i", "i2i"]:
            return self._load_i2v_data()
        else:
            return self._load_t2v_prompts()

    def _download_file(self, url: str, dest_path: str) -> None:
        """Download a file from URL to destination path."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        resp = requests.get(url)
        resp.raise_for_status()
        with open(dest_path, "w") as f:
            f.write(resp.text)

    def _load_t2v_prompts(self) -> list[dict[str, Any]]:
        path = self.args.dataset_path

        if not path:
            path = os.path.join(self.cache_dir, "vbench_subject_consistency.txt")
            if not os.path.exists(path):
                print(f"Downloading VBench T2V prompts to {path}...")
                try:
                    self._download_file(self.T2V_PROMPT_URL, path)
                except Exception as e:
                    print(f"Failed to download VBench prompts: {e}")
                    return [{"prompt": "A cat sitting on a bench"}] * 50

        prompts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({"prompt": line})

        return self._resize_data(prompts)

    def _auto_download_i2v_dataset(self) -> str:
        """Auto-download VBench I2V dataset and return the dataset directory."""
        vbench_i2v_dir = os.path.join(self.cache_dir, "vbench_i2v", "vbench2_beta_i2v")
        info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")

        if os.path.exists(info_json_path):
            return vbench_i2v_dir

        print(f"Downloading VBench I2V dataset to {vbench_i2v_dir}...")
        try:
            cache_root = os.path.join(self.cache_dir, "vbench_i2v")
            script_path = os.path.join(cache_root, "download_data.sh")

            self._download_file(self.I2V_DOWNLOAD_SCRIPT_URL, script_path)
            os.chmod(script_path, 0o755)

            print("Executing download_data.sh (this may take a while)...")
            import subprocess

            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Download script failed: {result.stderr}")

            print(f"Successfully downloaded VBench I2V dataset to {vbench_i2v_dir}")
        except Exception as e:
            print(f"Failed to download VBench I2V dataset: {e}")
            print("Please manually download following instructions at:")
            print("https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download")
            return None

        return vbench_i2v_dir if os.path.exists(info_json_path) else None

    def _load_from_i2v_json(self, json_path: str) -> list[dict[str, Any]]:
        """Load I2V data from i2v-bench-info.json format."""
        with open(json_path) as f:
            items = json.load(f)

        base_dir = os.path.dirname(os.path.dirname(json_path))  # Go up to vbench2_beta_i2v
        origin_dir = os.path.join(base_dir, "data", "origin")

        data = []
        for item in items:
            img_path = os.path.join(origin_dir, item.get("file_name", ""))
            if os.path.exists(img_path):
                data.append({"prompt": item.get("caption", ""), "image_path": img_path})
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(data)} I2V samples from VBench I2V dataset")
        return data

    def _scan_directory_for_images(self, path: str) -> list[dict[str, Any]]:
        """Scan directory for image files."""
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []

        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))

            # Also check in data/origin subdirectory
            origin_dir = os.path.join(path, "data", "origin")
            if os.path.exists(origin_dir):
                files.extend(glob.glob(os.path.join(origin_dir, ext)))
                files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

        return [{"prompt": os.path.splitext(os.path.basename(f))[0], "image_path": f} for f in files]

    def _create_dummy_data(self) -> list[dict[str, Any]]:
        """Create dummy data with a placeholder image in cache directory."""
        print("No I2V data found. Using dummy placeholders.")

        dummy_image = os.path.join(self.cache_dir, "dummy_image.jpg")
        if not os.path.exists(dummy_image):
            try:
                from PIL import Image

                os.makedirs(self.cache_dir, exist_ok=True)
                img = Image.new("RGB", (100, 100), color="red")
                img.save(dummy_image)
                print(f"Created dummy image at {dummy_image}")
            except ImportError:
                print("PIL not installed, cannot create dummy image.")
                return []

        return [{"prompt": "A moving cat", "image_path": dummy_image}] * 10

    def _load_i2v_data(self) -> list[dict[str, Any]]:
        """Load I2V data from VBench I2V dataset or user-provided path."""
        path = self.args.dataset_path

        # Auto-download if no path provided
        if not path:
            path = self._auto_download_i2v_dataset()
            if not path:
                return self._resize_data(self._create_dummy_data())

        # Try to load from i2v-bench-info.json
        info_json_candidates = [
            os.path.join(path, "data", "i2v-bench-info.json"),
            path if path.endswith(".json") else None,
        ]

        for json_path in info_json_candidates:
            if json_path and os.path.exists(json_path):
                try:
                    return self._resize_data(self._load_from_i2v_json(json_path))
                except Exception as e:
                    print(f"Failed to load {json_path}: {e}")

        # Fallback: scan directory for images
        if os.path.isdir(path):
            data = self._scan_directory_for_images(path)
            if data:
                return self._resize_data(data)

        # Last resort: dummy data
        return self._resize_data(self._create_dummy_data())

    def _resize_data(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resize data to match num_prompts."""
        if not self.args.num_prompts:
            return data

        if len(data) < self.args.num_prompts:
            factor = (self.args.num_prompts // len(data)) + 1
            data = data * factor

        return data[: self.args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        item = self.items[idx]
        image_paths = [item["image_path"]] if "image_path" in item else None

        return RequestFuncInput(
            prompt=item.get("prompt", ""),
            api_url=self.api_url,
            model=self.model,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            seed=self.args.seed,
            fps=self.args.fps,
            image_paths=image_paths,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class TraceDataset(BaseDataset):
    """Trace-based dataset loader for heterogeneous diffusion requests."""

    DEFAULT_REPO_ID = "asukaqaqzz/Dit_Trace"
    DEFAULT_FILENAME = "sd3_trace.txt"
    DEFAULT_FILENAME_BY_TASK: dict[str, str] = {
        # Text-to-image traces (e.g., SD3)
        "t2i": "sd3_trace.txt",
        # Text-to-video traces (e.g., CogVideoX)
        "t2v": "cogvideox_trace.txt",
    }

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm-omni", "trace")
        self.default_filename = self.DEFAULT_FILENAME_BY_TASK.get(getattr(args, "task", ""), self.DEFAULT_FILENAME)
        dataset_root = args.dataset_path
        if not dataset_root:
            dataset_root = self._download_default_trace()
        self.items = self._load_items(dataset_root)

    @staticmethod
    def _coerce_int(x: Any) -> int | None:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        try:
            s = str(x).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    @staticmethod
    def _coerce_float(x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, float):
            return x
        if isinstance(x, int):
            return float(x)
        try:
            s = str(x).strip()
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    def _download_default_trace(self) -> str:
        """Download default trace file from HuggingFace Hub if not provided."""

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download the default trace dataset. "
                "Install via `pip install huggingface_hub`."
            ) from exc

        os.makedirs(self.cache_dir, exist_ok=True)
        return hf_hub_download(
            repo_id=self.DEFAULT_REPO_ID,
            filename=self.default_filename,
            repo_type="dataset",
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False,
        )

    def _expand_paths(self, dataset_path: str | None) -> list[str]:
        if not dataset_path:
            return []

        parts = [p.strip() for p in str(dataset_path).split(",") if p.strip()]
        paths: list[str] = []
        for p in parts:
            if any(ch in p for ch in ["*", "?", "["]):
                paths.extend(sorted(glob.glob(p)))
            elif os.path.isdir(p):
                paths.extend(sorted(glob.glob(os.path.join(p, "**", "*.txt"), recursive=True)))
            else:
                paths.append(p)

        seen = set()
        unique_paths = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        return unique_paths

    def _parse_trace_file(self, path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def parse_request_repr_line(line: str) -> dict[str, Any] | None:
            text = line.strip()
            if not text:
                return None
            if not (text.startswith("Request(") and text.endswith(")")):
                return None
            inner = text[len("Request(") : -1]
            try:
                expr = ast.parse(f"f({inner})", mode="eval")
                if not isinstance(expr.body, ast.Call):
                    return None
                call = expr.body
                out: dict[str, Any] = {}
                for kw in call.keywords:
                    if kw.arg is None:
                        continue
                    out[kw.arg] = ast.literal_eval(kw.value)
                return out
            except Exception:
                return None

        # detect first non-empty line to pick parser
        first_non_empty = None
        with open(path, encoding="utf-8") as f:
            for _ in range(50):
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    first_non_empty = line.strip()
                    f.seek(pos)
                    break

        if first_non_empty is None:
            return rows

        if first_non_empty.startswith("Request("):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parsed = parse_request_repr_line(line)
                    if isinstance(parsed, dict):
                        rows.append(parsed)
            return rows

        # txt fallback: parse Request(...) lines only
        with open(path, encoding="utf-8") as f:
            for line in f:
                parsed = parse_request_repr_line(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return rows

    def _load_items(self, dataset_root: str) -> list[dict[str, Any]]:
        paths = self._expand_paths(dataset_root)
        if not paths:
            raise ValueError("No trace files found. Provide --dataset-path or rely on default HuggingFace download.")

        items: list[dict[str, Any]] = []
        for p in paths:
            if not os.path.exists(p):
                continue
            for row in self._parse_trace_file(p):
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("_source", p)
                    items.append(row)

        if not items:
            raise ValueError("Trace dataset is empty after parsing provided paths.")

        if self.args.num_prompts is not None:
            items = items[: self.args.num_prompts]

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        row = self.items[idx]
        prompt = row.get("prompt") or row.get("text") or ""

        row_height = self._coerce_int(row.get("height"))
        row_width = self._coerce_int(row.get("width"))
        num_frames = self._coerce_int(row.get("num_frames"))
        num_steps = self._coerce_int(row.get("num_inference_steps"))
        seed = self._coerce_int(row.get("seed"))
        fps = self._coerce_int(row.get("fps"))
        timestamp = self._coerce_float(row.get("timestamp"))
        slo_ms = self._coerce_float(row.get("slo_ms"))
        image_paths = row.get("image_paths")

        override_w = self.args.width
        override_h = self.args.height
        if override_w is not None or override_h is not None:
            width = override_w
            height = override_h
        else:
            width = row_width
            height = row_height

        return RequestFuncInput(
            prompt=str(prompt),
            api_url=self.api_url,
            model=self.model,
            width=width,
            height=height,
            num_frames=num_frames if num_frames is not None else self.args.num_frames,
            num_inference_steps=num_steps if num_steps is not None else self.args.num_inference_steps,
            seed=seed if seed is not None else self.args.seed,
            fps=fps if fps is not None else self.args.fps,
            timestamp=timestamp,
            slo_ms=slo_ms,
            image_paths=image_paths,
            request_id=str(row.get("request_id")) if row.get("request_id") is not None else str(uuid.uuid4()),
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class MixedPriorityDataset(BaseDataset):
    """Loader for the synthetic mixed-priority JSON workloads produced by
    ``benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py``.

    Each request carries:
      * shape (height, width, num_frames, num_steps, denoise_chunk_size)
      * absolute arrival_ms in workload time, mapped to RequestFuncInput.timestamp
        (in seconds, for ``iter_requests`` to schedule)
      * client_deadline_ms - arrival_ms -> slo_ms (relative latency budget the
        existing SLO machinery understands). Falls back to deadline_ms when the
        workload predates the client/scheduler deadline split. client_deadline_ms
        adds the fixed end-to-end tax (post-proc/IPC/poll) on top of the
        scheduler deadline, so the client SLO check is not charged for overhead
        the policy cannot control.
      * runtime_v2_priority / runtime_v2_deadline_ms / runtime_v2_arrival_ms /
        runtime_v2_request_class packed into a JSON ``extra_args`` form field
        so the server's edf_greedy / edf_best_fit policies can see them. The
        server's VideoGenerationRequest schema parses the JSON string back to
        a dict (see vllm_omni/entrypoints/openai/protocol/videos.py).
      * request_class on the input dataclass for per-class metrics
    """

    DEFAULT_PROMPTS_BY_CLASS: dict[str, str] = {
        "S": "a serene lakeside sunrise with mist over the water",
        "M": "a bustling city street at night with neon signs and people walking",
        "L": "a sweeping aerial shot over a mountain range at golden hour",
    }
    DEFAULT_PROMPT = "a placid scenic landscape in cinematic style"

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        path = args.workload_json or args.dataset_path
        if not path:
            raise ValueError(
                "MixedPriorityDataset requires --workload-json (or --dataset-path) "
                "pointing at a synthetic JSON workload"
            )
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict):
            self.metadata = doc.get("metadata", {}) or {}
            self.items: list[dict[str, Any]] = list(doc.get("requests", []) or [])
        elif isinstance(doc, list):
            self.metadata = {}
            self.items = list(doc)
        else:
            raise ValueError(f"workload JSON must be a dict or list, got {type(doc).__name__}")
        if not self.items:
            raise ValueError(f"workload {path!r} has no requests")
        if args.num_prompts is not None and args.num_prompts > 0:
            self.items = self.items[: args.num_prompts]

        # Optional override prompt
        self.override_prompt = getattr(args, "mixed_priority_prompt", None)
        # Normalize arrivals to start at 0 so timestamps render as offset-from-now.
        arrivals = [float(item.get("arrival_ms", 0.0) or 0.0) for item in self.items]
        self._arrival_zero_ms = min(arrivals) if arrivals else 0.0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        row = self.items[idx]
        request_class = str(row.get("request_class") or "M").upper()
        prompt = (
            self.override_prompt
            or self.DEFAULT_PROMPTS_BY_CLASS.get(request_class)
            or self.DEFAULT_PROMPT
        )
        arrival_ms = float(row.get("arrival_ms", 0.0) or 0.0)
        deadline_ms = row.get("deadline_ms")
        deadline_ms = float(deadline_ms) if deadline_ms is not None else None
        # Two distinct deadlines: deadline_ms is the SCHEDULER deadline (GPU
        # pipeline only) sent to the server's EDF policy; client_deadline_ms
        # additionally budgets the fixed end-to-end tax (video post-proc,
        # IPC, bench poll interval) the scheduler cannot control. The
        # client-side SLO check uses client_deadline_ms when present so it
        # measures scheduling quality, not measurement overhead.
        client_deadline_ms = row.get("client_deadline_ms")
        client_deadline_ms = (
            float(client_deadline_ms) if client_deadline_ms is not None else deadline_ms
        )
        slo_ms = (client_deadline_ms - arrival_ms) if client_deadline_ms is not None else None
        # Pack runtime_v2 metadata for the server's EDF/best-fit policy. JSON
        # string because v1/videos POST is multipart/form-data.
        ea: dict[str, Any] = {
            "runtime_v2_priority": int(row.get("priority", 0) or 0),
            "runtime_v2_request_class": request_class,
            "runtime_v2_arrival_ms": arrival_ms,
        }
        if deadline_ms is not None:
            ea["runtime_v2_deadline_ms"] = deadline_ms
        # Also surface latent_seq_len etc. for edf_best_fit's finish-time math
        # (server adapter may also fill these via plan.metadata, but client-side
        # is a no-cost shortcut).
        for k in ("latent_seq_len", "voxels", "num_steps", "text_seq_len", "denoise_chunk_size"):
            if k in row:
                ea[k] = row[k]
        extra_body = {"extra_args": json.dumps(ea, sort_keys=True)}
        # Optionally pin guidance_scale so a legacy server does CFG (its default
        # is 0.0 = no CFG, vs the runtime_v2 wan adapter's 4.0 fallback). Both
        # the v1/videos VideoGenerationRequest and the openai ImageGenerationRequest
        # expose a guidance_scale field, so this aligns either backend.
        gs = getattr(self.args, "guidance_scale", None)
        if gs is not None:
            # Set both experts (TI2V has high/low-noise) to match the runtime_v2
            # wan adapter, where guidance_high = guidance_scale_2 or guidance_low,
            # i.e. both default to 4.0 when the client omits guidance.
            extra_body["guidance_scale"] = float(gs)
            extra_body["guidance_scale_2"] = float(gs)

        return RequestFuncInput(
            prompt=prompt,
            api_url=self.api_url,
            model=self.model,
            width=int(row["width"]) if row.get("width") is not None else None,
            height=int(row["height"]) if row.get("height") is not None else None,
            num_frames=int(row["num_frames"]) if row.get("num_frames") is not None else None,
            num_inference_steps=int(row["num_steps"]) if row.get("num_steps") is not None else None,
            seed=self.args.seed,
            timestamp=(arrival_ms - self._arrival_zero_ms) / 1000.0,
            slo_ms=slo_ms,
            request_class=request_class,
            request_id=str(row.get("request_id") or f"mp_{idx:06d}"),
            extra_body=extra_body,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class RandomDataset(BaseDataset):
    def __init__(self, args, api_url: str, model: str, enable_negative_prompt: bool = False):
        super().__init__(args, api_url, model)
        self.num_prompts = args.num_prompts
        self.enable_negative_prompt = enable_negative_prompt
        self.random_request_config = getattr(args, "random_request_config", None)
        if self.random_request_config:
            self.random_request_config = json.loads(self.random_request_config)
            self._weights = [p["weight"] for p in self.random_request_config]

            self.random_request_config = [
                {k: v for k, v in p.items() if k != "weight"} for p in self.random_request_config
            ]

            seed = getattr(args, "random_request_seed", 42)
            self._rng = random.Random(seed)

            self._sampled_requests = self._rng.choices(
                self.random_request_config,
                weights=self._weights,
                k=self.num_prompts,
            )
        else:
            self._sampled_requests = None

        # Random image generate
        if self.args.task in ["i2v", "ti2v", "ti2i", "i2i"]:
            img = Image.new("RGB", (512, 512), (255, 255, 255))

            image_path = os.path.join(tempfile.gettempdir(), "diffusion_benchmark_random_image.png")
            self._random_image_path = [image_path]
            img.save(image_path)
        else:
            self._random_image_path = None

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> RequestFuncInput:
        extra_body = {}
        if self.enable_negative_prompt:
            extra_body["negative_prompt"] = f"Negative prompt {idx} for benchmarking diffusion models"

        params = {
            "width": self.args.width,
            "height": self.args.height,
            "num_frames": self.args.num_frames,
            "num_inference_steps": self.args.num_inference_steps,
            "fps": self.args.fps,
        }
        if self._sampled_requests:
            profile = self._sampled_requests[idx]
            params.update(profile)
        return RequestFuncInput(
            prompt=f"Random prompt {idx} for benchmarking diffusion models",
            api_url=self.api_url,
            model=self.model,
            seed=self.args.seed,
            extra_body=extra_body,
            image_paths=self._random_image_path,
            **params,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


def _compute_expected_latency_ms_from_base(req: RequestFuncInput, args, base_time_ms: float | None) -> float | None:
    """Compute expected execution time (ms) based on a base per-step-per-frame unit time.

    Assumes linear scaling with pixel area, frame count, and num_inference_steps.
    The base unit represents latency for a 16x16 resolution, single frame, single step.
    """

    if base_time_ms is None:
        return None

    width = req.width if req.width is not None else args.width
    height = req.height if req.height is not None else args.height
    if width is None or height is None:
        return None

    frames = req.num_frames if req.num_frames is not None else args.num_frames
    steps = req.num_inference_steps if req.num_inference_steps is not None else args.num_inference_steps

    frame_scale = frames if isinstance(frames, int) and frames > 0 else 1
    step_scale = steps if isinstance(steps, int) and steps > 0 else 1

    area_units = max((float(width) * float(height)) / float(16 * 16), 1.0)
    return float(base_time_ms) * area_units * frame_scale * step_scale


def _infer_slo_base_time_ms_from_warmups(
    warmup_pairs: list[tuple[RequestFuncInput, RequestFuncOutput]],
    args,
) -> float | None:
    """Infer base SLO unit time from warmup requests.

    Returns the median base latency (ms) for a 16x16 resolution, single-frame,
    single-step request. Only uses warmups that succeeded and have resolvable
    width/height.
    """

    candidates_ms: list[float] = []
    for req, out in warmup_pairs:
        if not out.success or out.latency <= 0:
            continue

        width = req.width if req.width is not None else args.width
        height = req.height if req.height is not None else args.height
        if width is None or height is None:
            continue

        frames = req.num_frames if req.num_frames is not None else args.num_frames
        steps = req.num_inference_steps if req.num_inference_steps is not None else args.num_inference_steps

        frame_scale = int(frames) if isinstance(frames, int) and frames > 0 else 1
        step_scale = int(steps) if isinstance(steps, int) and steps > 0 else 1

        area_units = max((float(width) * float(height)) / float(16 * 16), 1.0)
        denom = area_units * float(frame_scale) * float(step_scale)
        if denom <= 0:
            continue

        candidates_ms.append((out.latency * 1000.0) / denom)

    if not candidates_ms:
        return None
    return float(np.median(candidates_ms))


def _populate_slo_ms_from_warmups(
    requests_list: list[RequestFuncInput],
    warmup_pairs: list[tuple[RequestFuncInput, RequestFuncOutput]],
    args,
) -> list[RequestFuncInput]:
    """Populate missing RequestFuncInput.slo_ms using warmup outputs.

    - If a request already has slo_ms (e.g., trace-provided), it is kept as-is.
    - If any request has slo_ms is None and we can infer base time from warmups,
      we estimate each missing request's expected execution time and set:
        req.slo_ms = expected_latency_ms * args.slo_scale

    Returns updated requests_list.
    """

    if not any(req.slo_ms is None for req in requests_list):
        return requests_list

    base_time_ms = _infer_slo_base_time_ms_from_warmups(warmup_pairs, args)
    if base_time_ms is None:
        return requests_list

    slo_scale = float(getattr(args, "slo_scale", 3.0))
    if slo_scale <= 0:
        raise ValueError(f"slo_scale must be positive, got {slo_scale}.")

    updated: list[RequestFuncInput] = []
    for req in requests_list:
        if req.slo_ms is not None:
            updated.append(req)
            continue
        expected_ms = _compute_expected_latency_ms_from_base(req, args, base_time_ms)
        updated.append(replace(req, slo_ms=(expected_ms * slo_scale) if expected_ms is not None else None))

    return updated


async def iter_requests(
    requests_list: list[RequestFuncInput],
    request_rate: float,
    *,
    use_trace_arrivals: bool = False,
) -> AsyncGenerator[RequestFuncInput, None]:
    """Yield requests with one of three pacing modes.

    - ``use_trace_arrivals=True``: honor each request's ``timestamp`` (seconds
      from start of replay) and sleep until that offset before yielding. Used
      with MixedPriorityDataset; ``request_rate`` is ignored.
    - ``request_rate == inf``: yield all immediately, no sleep.
    - otherwise: fixed cadence ``1 / request_rate`` seconds between yields.
    """

    if use_trace_arrivals:
        start = asyncio.get_event_loop().time()
        for req in requests_list:
            target_offset = float(req.timestamp) if req.timestamp is not None else 0.0
            now_offset = asyncio.get_event_loop().time() - start
            delay = target_offset - now_offset
            if delay > 0:
                await asyncio.sleep(delay)
            yield req
        return

    if request_rate != float("inf"):
        if request_rate <= 0:
            raise ValueError(f"request_rate must be positive or inf, got {request_rate}.")
        interval_s = 1.0 / float(request_rate)

    for i, req in enumerate(requests_list):
        if request_rate != float("inf") and i > 0:
            await asyncio.sleep(interval_s)
        yield req


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    total_duration: float,
    requests_list: list[RequestFuncInput],
    args,
    slo_enabled: bool,
):
    success_outputs = [o for o in outputs if o.success]
    error_outputs = [o for o in outputs if not o.success]

    num_success = len(success_outputs)
    latencies = [o.latency for o in success_outputs]
    peak_memories = [o.peak_memory_mb for o in success_outputs if o.peak_memory_mb > 0]

    metrics = {
        "duration": total_duration,
        "completed_requests": num_success,
        "failed_requests": len(error_outputs),
        "throughput_qps": num_success / total_duration if total_duration > 0 else 0,
        "latency_mean": np.mean(latencies) if latencies else 0,
        "latency_median": np.median(latencies) if latencies else 0,
        "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        "latency_p95": np.percentile(latencies, 95) if latencies else 0,
        "latency_p50": np.percentile(latencies, 50) if latencies else 0,
        "peak_memory_mb_max": max(peak_memories) if peak_memories else 0,
        "peak_memory_mb_mean": np.mean(peak_memories) if peak_memories else 0,
        "peak_memory_mb_median": np.median(peak_memories) if peak_memories else 0,
    }

    if slo_enabled:
        slo_defined_total = 0
        slo_met_success = 0

        for req, out in zip(requests_list, outputs):
            if req.slo_ms is None:
                continue
            slo_defined_total += 1
            if out.slo_achieved is None:
                continue
            if out.slo_achieved:
                slo_met_success += 1

        slo_attain_all = (slo_met_success / slo_defined_total) if slo_defined_total > 0 else 0.0

        metrics.update(
            {
                "slo_attainment_rate": slo_attain_all,
                "slo_met_success": slo_met_success,
                "slo_scale": getattr(args, "slo_scale", 3.0),
            }
        )

    # Per-class breakdown (latency p95, SLO attainment per S/M/L). Driven by
    # RequestFuncInput.request_class; empty if no request has it set.
    by_class: dict[str, dict[str, Any]] = {}
    for req, out in zip(requests_list, outputs):
        klass = getattr(req, "request_class", None)
        if not klass:
            continue
        bucket = by_class.setdefault(
            klass,
            {"n": 0, "completed": 0, "latencies": [], "slo_defined": 0, "slo_met": 0},
        )
        bucket["n"] += 1
        if out.success:
            bucket["completed"] += 1
            bucket["latencies"].append(out.latency)
        if req.slo_ms is not None:
            bucket["slo_defined"] += 1
            if out.slo_achieved:
                bucket["slo_met"] += 1
    if by_class:
        per_class: dict[str, dict[str, float]] = {}
        for klass, bucket in sorted(by_class.items()):
            lats = bucket["latencies"]
            per_class[klass] = {
                "n": bucket["n"],
                "completed": bucket["completed"],
                "latency_mean": float(np.mean(lats)) if lats else 0.0,
                "latency_p50": float(np.percentile(lats, 50)) if lats else 0.0,
                "latency_p95": float(np.percentile(lats, 95)) if lats else 0.0,
                "latency_p99": float(np.percentile(lats, 99)) if lats else 0.0,
                "slo_defined": bucket["slo_defined"],
                "slo_met": bucket["slo_met"],
                "slo_attainment_rate": (
                    bucket["slo_met"] / bucket["slo_defined"]
                    if bucket["slo_defined"] > 0
                    else 0.0
                ),
            }
        metrics["per_class"] = per_class

    # Per-request dump for offline lateness analysis. lateness_ms is the
    # client-side overshoot (latency - slo budget); positive => violated. The
    # client deadline_const tax cancels (it is in both latency and slo_ms), so
    # this equals the scheduler-side overshoot too.
    per_request: list[dict[str, Any]] = []
    for req, out in zip(requests_list, outputs):
        slo_ms = getattr(req, "slo_ms", None)
        lat_ms = (out.latency * 1000.0) if out.success else None
        lateness_ms = (
            (lat_ms - slo_ms) if (lat_ms is not None and slo_ms is not None) else None
        )
        per_request.append(
            {
                "request_id": getattr(req, "request_id", None),
                "request_class": getattr(req, "request_class", None),
                "success": bool(out.success),
                "latency_ms": lat_ms,
                "slo_ms": slo_ms,
                "slo_achieved": out.slo_achieved,
                "lateness_ms": lateness_ms,
            }
        )
    metrics["per_request"] = per_request

    return metrics


def wait_for_service(base_url: str, timeout: int = 120) -> None:
    print(f"Waiting for service at {base_url}...")
    start_time = time.time()
    while True:
        try:
            # Try /health endpoint first
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                print("Service is ready.")
                break
        except requests.exceptions.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Service at {base_url} did not start within {timeout} seconds.")

        time.sleep(1)


async def benchmark(args):
    # Construct base_url if not provided
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    VIDEO_TASKS = {"t2v", "i2v", "ti2v"}
    IMAGE_TASKS = {"t2i", "i2i", "ti2i"}

    if args.task in VIDEO_TASKS:
        task_type = "2v"
    elif args.task in IMAGE_TASKS:
        task_type = "2i"
    else:
        raise ValueError(
            f"Unsupported task: '{args.task}'. "
            f"Valid video tasks: {sorted(VIDEO_TASKS)}, "
            f"Valid image tasks: {sorted(IMAGE_TASKS)}"
        )

    if args.debug_video_requests:
        os.environ["VLLM_OMNI_BENCH_VIDEO_DEBUG"] = "1"
        print("Enabled detailed /v1/videos benchmark request logging.")
    else:
        os.environ.pop("VLLM_OMNI_BENCH_VIDEO_DEBUG", None)

    valid_backends = sorted(backends_function_mapping[task_type].keys())

    if args.backend not in valid_backends:
        logger.error(
            f"Invalid backend '{args.backend}' for task '{args.task}' (task type: '{task_type}').\n"
            f"Valid backends for this task type: {valid_backends}\n"
            f"Example usage: --task {args.task} --backend {valid_backends[0]}"
        )
        raise ValueError("Backend validation failed. See log above for valid options.")

    # Setup API URL and request function based on backend
    request_func, api_url = backends_function_mapping[task_type][args.backend]
    api_url = f"{args.base_url}{api_url}"

    if args.dataset == "vbench":
        dataset = VBenchDataset(args, api_url, args.model)
    elif args.dataset == "trace":
        dataset = TraceDataset(args, api_url, args.model)
    elif args.dataset == "random":
        dataset = RandomDataset(args, api_url, args.model, args.enable_negative_prompt)
    elif args.dataset == "mixed_priority":
        dataset = MixedPriorityDataset(args, api_url, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("Loading requests...")
    requests_list = dataset.get_requests()
    requests_list = [replace(req, video_job_timeout_seconds=args.video_job_timeout_seconds) for req in requests_list]
    print(f"Prepared {len(requests_list)} requests from {args.dataset} dataset.")

    # Limit concurrency
    if args.max_concurrency is not None:
        semaphore = asyncio.Semaphore(args.max_concurrency)
    else:
        semaphore = None

    async def limited_request_func(req, session, pbar):
        if semaphore:
            async with semaphore:
                return await request_func(req, session, pbar)
        else:
            return await request_func(req, session, pbar)

    # Run benchmark
    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)

    async with aiohttp.ClientSession() as session:
        warmup_pairs: list[tuple[RequestFuncInput, RequestFuncOutput]] = []
        if args.warmup_requests and requests_list:
            print(
                f"Running {args.warmup_requests} warmup request(s) \
                with num_inference_steps={args.warmup_num_inference_steps}..."
            )
            for i in range(args.warmup_requests):
                warm_req = requests_list[i % len(requests_list)]
                if args.warmup_num_inference_steps is not None:
                    warm_req = replace(
                        warm_req,
                        num_inference_steps=args.warmup_num_inference_steps,
                    )
                warm_out = await limited_request_func(warm_req, session, None)
                warmup_pairs.append((warm_req, warm_out))

        if args.slo:
            # Prefer trace-provided per-request slo_ms. Only populate when missing.
            requests_list = _populate_slo_ms_from_warmups(
                requests_list=requests_list,
                warmup_pairs=warmup_pairs,
                args=args,
            )

        start_time = time.perf_counter()
        tasks = []
        # Mixed-priority workloads encode per-request arrival_ms; replay the
        # trace by sleeping to each timestamp instead of using --request-rate.
        use_trace_arrivals = (
            args.dataset == "mixed_priority"
            and any(req.timestamp is not None for req in requests_list)
        )
        async for req in iter_requests(
            requests_list=requests_list,
            request_rate=args.request_rate,
            use_trace_arrivals=use_trace_arrivals,
        ):
            task = asyncio.create_task(limited_request_func(req, session, pbar))
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    pbar.close()

    # Calculate metrics
    metrics = calculate_metrics(outputs, total_duration, requests_list, args, args.slo)

    # Add configuration info to metrics for JSON output
    metrics["backend"] = args.backend
    metrics["model"] = args.model
    metrics["dataset"] = args.dataset
    metrics["task"] = args.task

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=60, c="="))

    # Section 1: Configuration
    print("{:<40} {:<15}".format("Backend:", args.backend))
    print("{:<40} {:<15}".format("Model:", args.model))
    print("{:<40} {:<15}".format("Dataset:", args.dataset))
    print("{:<40} {:<15}".format("Task:", args.task))

    # Section 2: Execution & Traffic
    print(f"{'-' * 50}")
    print("{:<40} {:<15.2f}".format("Benchmark duration (s):", metrics["duration"]))
    print("{:<40} {:<15}".format("Request rate:", str(args.request_rate)))
    print(
        "{:<40} {:<15}".format(
            "Max request concurrency:",
            str(args.max_concurrency) if args.max_concurrency else "not set",
        )
    )
    print("{:<40} {}/{:<15}".format("Successful requests:", metrics["completed_requests"], len(requests_list)))

    # Section 3: Performance Metrics
    print(f"{'-' * 50}")

    print("{:<40} {:<15.2f}".format("Request throughput (req/s):", metrics["throughput_qps"]))
    print("{:<40} {:<15.4f}".format("Latency Mean (s):", metrics["latency_mean"]))
    print("{:<40} {:<15.4f}".format("Latency Median (s):", metrics["latency_median"]))
    print("{:<40} {:<15.4f}".format("Latency P99 (s):", metrics["latency_p99"]))
    print("{:<40} {:<15.4f}".format("Latency P95 (s):", metrics["latency_p95"]))

    if args.slo:
        print(f"{'-' * 50}")
        print("{:<40} {:<15.2%}".format("SLO Attainment Rate (all):", metrics.get("slo_attainment_rate", 0.0)))
        print("{:<40} {:<15}".format("SLO Met (success count):", str(metrics.get("slo_met_success", 0))))
        print("{:<40} {:<15}".format("SLO Scale:", str(metrics.get("slo_scale", 3.0))))

    if metrics["peak_memory_mb_max"] > 0:
        print(f"{'-' * 50}")
        print("{:<40} {:<15.2f}".format("Peak Memory Max (MB):", metrics["peak_memory_mb_max"]))
        print("{:<40} {:<15.2f}".format("Peak Memory Mean (MB):", metrics["peak_memory_mb_mean"]))
        print("{:<40} {:<15.2f}".format("Peak Memory Median (MB):", metrics["peak_memory_mb_median"]))

    print("\n" + "=" * 60)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark serving for diffusion models.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of the server (e.g., http://localhost:8091). Overrides host/port.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8091, help="Server port.")
    parser.add_argument("--model", type=str, default="default", help="Model name.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm-omni",
        choices=["vllm-omni", "openai", "v1/videos"],
        help="Backend to target the benchmark to.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vbench",
        choices=["vbench", "trace", "random", "mixed_priority"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--workload-json",
        type=str,
        default=None,
        help=(
            "Path to a synthetic mixed-priority workload JSON (produced by "
            "generate_runtime_v2_mixed_priority_workload.py). Used with "
            "--dataset mixed_priority. Per-request arrival_ms gates dispatch; "
            "slo_ms = deadline_ms - arrival_ms is computed client-side. "
            "runtime_v2_priority/deadline_ms/arrival_ms are packed into an "
            "extra_args JSON form field so the server's edf_greedy/edf_best_fit "
            "policies can read them."
        ),
    )
    parser.add_argument(
        "--mixed-priority-prompt",
        type=str,
        default=None,
        help=(
            "Override prompt for every mixed-priority request. Default behavior "
            "picks a static per-class prompt (S/M/L). Has no effect on output "
            "quality (prompts are not graded) but pins prompt length for stable "
            "latency comparisons."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "ti2v", "ti2i", "i2i", "t2i"],
        help="Task type.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset file (optional).",
    )
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to benchmark.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="If set, inject this guidance_scale into every request body (mixed_priority "
        "dataset). Needed to align a legacy server (whose default guidance_scale is 0.0 = "
        "no CFG) with the runtime_v2 wan adapter, which falls back to default_guidance_scale "
        "=4.0 (CFG on) when the client omits guidance. Sent as a form field for v1/videos "
        "and in the JSON payload for the openai image backend.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests, default to `1`. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of warmup requests to run before measurement.",
    )
    parser.add_argument(
        "--warmup-num-inference-steps",
        type=int,
        default=1,
        help="num_inference_steps used for warmup requests.",
    )
    parser.add_argument("--width", type=int, default=None, help="Image/Video width.")
    parser.add_argument("--height", type=int, default=None, help="Image/Video height.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames (for video).")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps (for diffusion models).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for diffusion models).",
    )
    parser.add_argument("--fps", type=int, default=None, help="FPS (for video).")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file for metrics.")
    parser.add_argument(
        "--slo",
        action="store_true",
        help=(
            "Enable SLO calculation and reporting. If trace provides per-request slo_ms, it is used. "
            "Otherwise, warmup request(s) are used to infer expected execution time assuming linear "
            "scaling by resolution, frames, and steps, then slo_ms = expected_time * --slo-scale."
        ),
    )
    parser.add_argument(
        "--slo-scale",
        type=float,
        default=3.0,
        help="SLO target multiplier: slo_ms = estimated_exec_time_ms * slo_scale (default: 3).",
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")
    parser.add_argument(
        "--debug-video-requests",
        action="store_true",
        default=False,
        help=(
            "Enable detailed client-side logs for /v1/videos requests "
            "(POST/poll/content/DELETE with client request_id and job_id mapping)."
        ),
    )
    parser.add_argument(
        "--video-job-timeout-seconds",
        type=float,
        default=3600.0,
        help=(
            "Client-side timeout in seconds while polling a /v1/videos job until completion "
            "(default: 3600). Only applies to --backend v1/videos."
        ),
    )
    parser.add_argument(
        "--enable-negative-prompt",
        action="store_true",
        default=False,
        help="Generate negative prompts when using the random dataset.",
    )
    parser.add_argument(
        "--random-request-config",
        type=str,
        default=None,
        help=(
            "JSON string defining random request profiles. "
            "Each profile may contain: width, height, num_inference_steps, etc. "
            "The 'weight' field controls sampling probability (relative weight). "
            "Example: "
            '[{"width":512,"height":512,"num_inference_steps":20,"weight":0.15},'
            '{"width":768,"height":768,"num_inference_steps":20,"weight":0.85}]'
        ),
    )

    args = parser.parse_args()

    asyncio.run(benchmark(args))
