"""Configurable Hugging Face and Mistral answer generation."""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "huggingface").strip().lower()


def configured_model() -> str:
    if provider_name() == "mistral":
        return os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")


def _huggingface(prompt: str) -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    model = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")
    if not token:
        raise RuntimeError("HF_TOKEN is required when LLM_PROVIDER=huggingface.")
    url = os.getenv("HF_API_URL", f"https://router.huggingface.co/hf-inference/models/{model}")
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 160, "return_full_text": False}},
        timeout=90,
    )
    response.raise_for_status()
    result: Any = response.json()
    if isinstance(result, list) and result and "generated_text" in result[0]:
        return str(result[0]["generated_text"])
    if isinstance(result, dict) and "generated_text" in result:
        return str(result["generated_text"])
    raise RuntimeError(f"Unexpected Hugging Face response: {result}")


def _mistral(prompt: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is required when LLM_PROVIDER=mistral.")
    response = requests.post(
        os.getenv("MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions"),
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 160,
        },
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()
    return str(result["choices"][0]["message"]["content"])


def generate_answer(prompt: str) -> str:
    """Generate with the configured provider."""
    if provider_name() == "mistral":
        return _mistral(prompt)
    if provider_name() in {"huggingface", "hf"}:
        return _huggingface(prompt)
    raise RuntimeError("Unsupported LLM_PROVIDER. Choose 'huggingface' or 'mistral'.")
