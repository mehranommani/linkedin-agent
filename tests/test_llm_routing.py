"""
Unit tests for LLM routing, provider fallback, and post guardrails.
Run with: PYTHONPATH=. pytest tests/ -v
"""
import asyncio
import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_groq_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _rate_limit_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 429
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429 Too Many Requests", request=MagicMock(), response=resp
    )
    return resp


# ── Provider loading ──────────────────────────────────────────────────────────

def test_providers_loaded_from_env(tmp_path, monkeypatch):
    """Keys in .env are picked up and become separate providers."""
    env = tmp_path / ".env"
    env.write_text(
        "GROQ_API_KEY_1=key_aaa\n"
        "GROQ_API_KEY_2=key_bbb\n"
    )
    from backend.mcp.tools import llm as llm_mod
    with patch.object(llm_mod, "_load_groq_keys_from_env", return_value=["key_aaa", "key_bbb"]):
        providers = llm_mod._get_providers()
    groq_providers = [p for p in providers if p["name"].startswith("groq")]
    assert len(groq_providers) == 2
    assert groq_providers[0]["api_key"] == "key_aaa"
    assert groq_providers[1]["api_key"] == "key_bbb"


def test_ollama_always_last_fallback():
    """Ollama is always the last entry in the provider list."""
    from backend.mcp.tools import llm as llm_mod
    providers = llm_mod._get_providers()
    assert providers[-1]["name"] == "ollama"
    assert providers[-1]["api_url"] is None


def test_real_env_keys_loaded():
    """Confirm the 3 real Groq keys are found in the project .env file."""
    from backend.mcp.tools.llm import _load_groq_keys_from_env
    keys = _load_groq_keys_from_env()
    assert len(keys) == 3, f"Expected 3 keys, got {len(keys)}"
    for key in keys:
        assert key.startswith("gsk_"), f"Key doesn't look like a Groq key: {key[:10]}"


# ── Fallback on 429 ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_to_second_key_on_rate_limit():
    """When provider 1 returns 429, provider 2 is tried and succeeds."""
    from backend.mcp.tools import llm as llm_mod

    providers = [
        {"name": "groq-1", "api_url": "https://api.groq.com/openai/v1", "api_key": "key1",
         "model": "llama-3.1-8b-instant", "judge_model": "llama-3.3-70b-versatile"},
        {"name": "groq-2", "api_url": "https://api.groq.com/openai/v1", "api_key": "key2",
         "model": "llama-3.1-8b-instant", "judge_model": "llama-3.3-70b-versatile"},
        {"name": "ollama", "api_url": None, "api_key": "", "model": "qwen2.5:7b", "judge_model": "qwen2.5:7b"},
    ]

    call_count = 0
    async def mock_post(url, json=None, headers=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _rate_limit_response()
        return _make_groq_response("Hello from key2")

    mock_client = AsyncMock()
    mock_client.post = mock_post

    with patch.object(llm_mod, "_get_providers", return_value=providers), \
         patch.object(llm_mod, "_get_backend", return_value="groq"), \
         patch.object(llm_mod, "_get_client", return_value=mock_client):
        result = await llm_mod._chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            json_mode=False,
            use_judge=False,
        )

    assert result == "Hello from key2"
    assert call_count == 2


@pytest.mark.asyncio
async def test_all_cloud_fail_falls_back_to_ollama():
    """When all cloud keys are rate-limited, local Ollama is used."""
    from backend.mcp.tools import llm as llm_mod

    providers = [
        {"name": "groq-1", "api_url": "https://api.groq.com/openai/v1", "api_key": "k1",
         "model": "llama-3.1-8b-instant", "judge_model": "llama-3.3-70b-versatile"},
        {"name": "ollama", "api_url": None, "api_key": "", "model": "qwen2.5:7b", "judge_model": "qwen2.5:7b"},
    ]

    async def mock_post_rate_limit(url, json=None, headers=None, **kwargs):
        return _rate_limit_response()

    mock_client = AsyncMock()
    mock_client.post = mock_post_rate_limit

    ollama_response = {"message": {"content": "Ollama fallback response"}}

    async def mock_ollama_request(url, payload, retries=2):
        return ollama_response

    with patch.object(llm_mod, "_get_providers", return_value=providers), \
         patch.object(llm_mod, "_get_backend", return_value="groq"), \
         patch.object(llm_mod, "_get_client", return_value=mock_client), \
         patch.object(llm_mod, "_ollama_request", side_effect=mock_ollama_request):
        result = await llm_mod._chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            json_mode=False,
            use_judge=False,
        )

    assert result == "Ollama fallback response"


# ── Guardrails ────────────────────────────────────────────────────────────────

def test_guardrail_no_code():
    from backend.evaluation.metrics import check_programmatic_guardrails
    post = "A" * 950 + " http://example.com\n#AI #ML #LLM #Agents"
    # inject code
    bad_post = post + "\ndef my_func():\n    pass"
    result = check_programmatic_guardrails(bad_post)
    issues = result["issues"]
    assert any("code" in i.lower() or "programming" in i.lower() for i in issues), issues


def test_guardrail_markdown_badges():
    from backend.evaluation.metrics import check_programmatic_guardrails
    post = "A" * 950 + " http://example.com\n#AI #ML #LLM #Agents"
    bad_post = post + "\n[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)"
    result = check_programmatic_guardrails(bad_post)
    assert any("badge" in i.lower() or "markdown" in i.lower() for i in result["issues"])


def test_guardrail_text_after_hashtags():
    from backend.evaluation.metrics import check_programmatic_guardrails
    bad_post = "A" * 950 + " http://example.com\n#AI #ML #LLM #Agents\nSome extra text here"
    result = check_programmatic_guardrails(bad_post)
    assert any("after hashtags" in i for i in result["issues"])


def test_guardrail_star_bullets():
    from backend.evaluation.metrics import check_programmatic_guardrails
    bad_post = "* Feature one\n* Feature two\n" + "A" * 900 + " http://example.com\n#AI #ML #LLM #Agents"
    result = check_programmatic_guardrails(bad_post)
    assert any("bullet" in i.lower() or "star" in i.lower() for i in result["issues"])


def test_guardrail_clean_post_passes():
    from backend.evaluation.metrics import check_programmatic_guardrails
    good_post = (
        "AI agents are changing how we build software.\n\n"
        "ClawTeam from HKUDS uses swarm intelligence to automate complex tasks "
        "with a single command. Instead of configuring dozens of steps manually, "
        "you describe the goal and the system figures out the path.\n\n"
        "This is the kind of tooling that lets small teams punch above their weight. "
        "The architecture is elegant — each agent specializes in one thing, and a "
        "coordinator routes tasks to the right specialist automatically.\n\n"
        "What makes this particularly interesting is the emergent behavior you get "
        "when multiple agents collaborate. Problems that would take a single agent "
        "many retries get solved in one pass because the right expert handles each step.\n\n"
        "This is the direction agentic AI is heading — not one massive model trying "
        "to do everything, but specialized agents working in concert. Think of it "
        "like a well-run engineering team versus one overloaded generalist.\n\n"
        "What tasks would you automate first if you had a swarm of specialized agents?\n\n"
        "https://github.com/HKUDS/ClawTeam\n\n"
        "#AI #AgenticAI #MachineLearning #OpenSource"
    )
    result = check_programmatic_guardrails(good_post)
    assert result["passed"], f"Expected clean post to pass, got issues: {result['issues']}"


def test_guardrail_missing_url():
    from backend.evaluation.metrics import check_programmatic_guardrails
    bad_post = "A" * 950 + "\n#AI #ML #LLM #Agents"
    result = check_programmatic_guardrails(bad_post)
    assert any("url" in i.lower() for i in result["issues"])
