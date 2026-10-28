#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send one exact-token-count decode-focused request to a running vLLM server."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST {path} failed: {exc}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"POST {path} returned non-JSON response: {body[:500]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"POST {path} returned unexpected JSON type: {type(value).__name__}")
    return value


def exact_prompt_token_ids(base_url: str, model: str, target_tokens: int, timeout: int) -> list[int]:
    phrase = "DSA resident capacity observation baseline. "
    repetitions = max(64, target_tokens // 3)
    for _ in range(10):
        payload = {
            "model": model,
            "prompt": phrase * repetitions,
            "add_special_tokens": False,
        }
        result = post_json(base_url, "/tokenize", payload, timeout)
        tokens = result.get("tokens")
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            if len(tokens) == 1:
                tokens = tokens[0]
        if isinstance(tokens, list):
            try:
                token_ids = [int(token) for token in tokens]
            except (TypeError, ValueError):
                token_ids = []
            if len(token_ids) >= target_tokens:
                return token_ids[:target_tokens]
        repetitions *= 2
    raise RuntimeError(
        "The /tokenize endpoint did not return enough integer token IDs. "
        "Check the vLLM /tokenize schema and the served model name."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="glm", help="Served model name, not filesystem path")
    parser.add_argument("--input-tokens", type=int, default=12000)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--trace-dir", required=True, help="Same directory exported by the server launcher")
    args = parser.parse_args()

    if args.input_tokens <= 0 or args.output_tokens <= 0:
        parser.error("input and output token counts must be positive")

    trace_dir = Path(args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()

    token_ids = exact_prompt_token_ids(
        args.base_url, args.model, args.input_tokens, args.timeout
    )
    request_meta = {
        "model": args.model,
        "base_url": args.base_url,
        "requested_prompt_tokens": args.input_tokens,
        "requested_completion_tokens": args.output_tokens,
        "temperature": 0,
        "seed": args.seed,
        "created_unix_ns": started_ns,
        "prompt_transport": "token_ids_from_vllm_tokenize_endpoint",
    }
    (trace_dir / "client_request.json").write_text(
        json.dumps(request_meta, indent=2) + "\n", encoding="utf-8"
    )

    payload = {
        "model": args.model,
        "prompt": token_ids,
        "max_tokens": args.output_tokens,
        "min_tokens": args.output_tokens,
        "temperature": 0,
        "seed": args.seed,
        "stream": False,
        "n": 1,
        "ignore_eos": True,
    }
    response = post_json(args.base_url, "/v1/completions", payload, args.timeout)
    (trace_dir / "client_response.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    print(json.dumps({
        "status": "PASS" if (
            prompt_tokens == args.input_tokens
            and completion_tokens == args.output_tokens
        ) else "TOKEN_COUNT_MISMATCH",
        "requested_prompt_tokens": args.input_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "requested_completion_tokens": args.output_tokens,
        "actual_completion_tokens": completion_tokens,
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        "trace_dir": str(trace_dir),
        "request_id": response.get("id"),
    }, indent=2, ensure_ascii=False))

    if prompt_tokens != args.input_tokens or completion_tokens != args.output_tokens:
        print("ERROR: request token counts differ from requested counts", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
