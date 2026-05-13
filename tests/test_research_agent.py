"""
Research Agent Tests
====================
Tests for Phase 3: ResearchBrief model, content type classifier,
research prompts, research guardrails, research evaluation, pipeline
nodes, edge routing, and end-to-end integration.

Run with: PYTHONPATH=. pytest tests/test_research_agent.py -v
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# 1. ResearchBrief model validation
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchBriefModel:
    def _make_brief(self, **overrides):
        from backend.models import ResearchBrief
        defaults = dict(
            content_type="github_tool",
            angle="LMCache eliminates KV cache recomputation across requests, saving 50% compute",
            evidence=["15x throughput gain in multi-round QA", "3-10x TTFT reduction"],
            contrast="Before: cache dies when request ends. After: offloaded to shared CPU/disk layer.",
            limitations=["Requires vLLM or SGLang — no standalone mode"],
            hook_recommendation="curiosity-gap",
            hook_draft="Your LLM inference is burning 50% of its compute on work it has already done.",
            confidence=0.9,
            data_gaps=["no latency figures for disk-offload path"],
        )
        defaults.update(overrides)
        return ResearchBrief(**defaults)

    def test_valid_brief_constructs(self):
        brief = self._make_brief()
        assert brief.confidence == 0.9
        assert brief.hook_recommendation == "curiosity-gap"
        assert len(brief.evidence) == 2

    def test_confidence_clamped_to_range(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._make_brief(confidence=1.5)
        with pytest.raises(ValidationError):
            self._make_brief(confidence=-0.1)

    def test_invalid_content_type_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._make_brief(content_type="unknown_type")

    def test_invalid_hook_recommendation_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._make_brief(hook_recommendation="question-hook")  # not in Literal

    def test_all_content_types_accepted(self):
        for ct in ("github_tool", "github_paper_impl", "research_paper", "article_opinion", "announcement"):
            b = self._make_brief(content_type=ct)
            assert b.content_type == ct

    def test_all_hook_recommendations_accepted(self):
        for h in ("curiosity-gap", "pattern-interrupt", "bold-statement", "discovery", "story-hook", "contrarian"):
            b = self._make_brief(hook_recommendation=h)
            assert b.hook_recommendation == h


# ─────────────────────────────────────────────────────────────────────────────
# 2. Content type classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestContentTypeClassifier:
    def _classify(self, source, title="", url="", content=""):
        from backend.mcp.resources.prompts import _classify_content_type
        return _classify_content_type(source=source, title=title, url=url, content=content)

    def test_github_default_is_tool(self):
        assert self._classify("github", "LMCache", "https://github.com/lmcache", "KV cache library") == "github_tool"

    def test_github_with_paper_signal_is_paper_impl(self):
        assert self._classify("github", "TabPFN", "", "we propose a new neural network for tabular data") == "github_paper_impl"
        assert self._classify("github", "GraphRAG", "", "arxiv paper: entity resolution") == "github_paper_impl"
        assert self._classify("github", "Method", "", "Our approach achieves state of the art") == "github_paper_impl"

    def test_arxiv_is_research_paper(self):
        assert self._classify("arxiv", "Attention Is All You Need") == "research_paper"

    def test_papers_is_research_paper(self):
        assert self._classify("papers", "RLHF Survey") == "research_paper"

    def test_rss_with_announce_title_is_announcement(self):
        assert self._classify("rss", "Introducing GPT-5", "https://openai.com/blog") == "announcement"
        assert self._classify("rss", "Announcing the release of Claude 4") == "announcement"

    def test_rss_opinion_is_article_opinion(self):
        assert self._classify("rss", "The future of open-source AI", "https://example.com/post") == "article_opinion"

    def test_hackernews_opinion(self):
        assert self._classify("hackernews", "Why RAG pipelines fail in production") == "article_opinion"

    def test_fallback_is_github_tool(self):
        assert self._classify("unknown_source", "SomeTool") == "github_tool"

    def test_blog_url_signals_announcement(self):
        # A non-rss source with "blog" in URL → announcement
        assert self._classify("devto", "New model released", "https://company.com/blog/model") == "announcement"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Research prompts: completeness and content rules
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchPrompts:
    def _get_prompts(self, content_type, title="TestTool", content="content", url="https://example.com", issues=None):
        from backend.mcp.resources.prompts import research_prompt
        return research_prompt(content_type=content_type, title=title, content=content,
                               source_url=url, previous_issues=issues)

    def _all_types(self):
        return ["github_tool", "github_paper_impl", "research_paper", "article_opinion", "announcement"]

    def test_all_types_return_system_and_user(self):
        for ct in self._all_types():
            sys_p, usr_p = self._get_prompts(ct)
            assert len(sys_p) > 50
            assert len(usr_p) > 100

    def test_system_prompt_is_same_for_all_types(self):
        """System prompt (researcher persona) is shared across all content types."""
        prompts = [self._get_prompts(ct)[0] for ct in self._all_types()]
        assert all(p == prompts[0] for p in prompts), "System prompt differs between types"

    def test_github_tool_has_9_required_fields(self):
        _, usr_p = self._get_prompts("github_tool")
        for field in ["PROBLEM:", "EVIDENCE:", "CONTRAST:", "LIMITATIONS:",
                      "ANGLE:", "HOOK_RECOMMENDATION:", "HOOK_DRAFT:", "CONFIDENCE:", "DATA_GAPS:"]:
            assert field in usr_p, f"Missing field {field} in github_tool prompt"

    def test_research_paper_has_paper_specific_fields(self):
        _, usr_p = self._get_prompts("research_paper")
        assert "CLAIM:" in usr_p or "PROBLEM:" in usr_p
        assert "BASELINE:" in usr_p or "EVIDENCE:" in usr_p

    def test_article_opinion_has_dissent_field(self):
        """article_opinion must include a LIMITATIONS/counterargument check to prevent bias failures."""
        _, usr_p = self._get_prompts("article_opinion")
        assert "counterargument" in usr_p.lower() or "LIMITATIONS:" in usr_p

    def test_system_prompt_forbids_invention(self):
        sys_p, _ = self._get_prompts("github_tool")
        assert "Do NOT invent" in sys_p or "DO NOT invent" in sys_p

    def test_system_prompt_has_sentinel_instructions(self):
        sys_p, _ = self._get_prompts("github_tool")
        assert "[not stated]" in sys_p
        assert "[no benchmark data]" in sys_p

    def test_what_if_banned_in_system_prompt(self):
        sys_p, _ = self._get_prompts("github_tool")
        assert "What if you could" in sys_p  # appears as a forbidden example

    def test_previous_issues_injected(self):
        _, usr_p = self._get_prompts("github_tool", issues=["angle is generic", "no concrete evidence"])
        assert "PREVIOUS ATTEMPT FAILED" in usr_p
        assert "angle is generic" in usr_p

    def test_content_truncated_to_8000(self):
        long_content = "x" * 10000
        _, usr_p = self._get_prompts("github_tool", content=long_content)
        # Full 10k string must not appear; truncated 8000-char slice must appear
        assert "x" * 10000 not in usr_p
        assert "x" * 8000 in usr_p

    def test_schema_instruction_in_user_prompt(self):
        _, usr_p = self._get_prompts("github_tool")
        assert "ResearchBrief" in usr_p or "JSON" in usr_p


# ─────────────────────────────────────────────────────────────────────────────
# 4. Programmatic research guardrails
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchGuardrails:
    def _check(self, **kwargs):
        from backend.evaluation.evaluator import check_research_guardrails
        defaults = dict(
            angle="LMCache eliminates KV cache recomputation — saves 50% compute per request",
            evidence=["15x throughput gain", "3-10x TTFT reduction"],
            hook_draft="Your LLM inference burns 50% of its compute on repeated work.",
            confidence=0.85,
            content_type="github_tool",
        )
        defaults.update(kwargs)
        return check_research_guardrails(defaults)

    def test_good_brief_passes(self):
        passed, issues = self._check()
        assert passed, f"Expected pass, got issues: {issues}"

    def test_empty_angle_fails(self):
        passed, issues = self._check(angle="")
        assert not passed
        assert any("angle" in i.lower() for i in issues)

    def test_generic_angle_fails(self):
        for generic in ["a powerful tool for developers", "it makes it easier to work with models",
                        "designed to improve performance"]:
            passed, issues = self._check(angle=generic)
            assert not passed, f"Generic angle '{generic}' should fail"
            assert any("generic" in i.lower() for i in issues)

    def test_all_sentinel_evidence_fails(self):
        passed, issues = self._check(evidence=["[no benchmark data]", "[not stated]"])
        assert not passed
        assert any("evidence" in i.lower() for i in issues)

    def test_one_real_evidence_passes(self):
        passed, _ = self._check(evidence=["[no benchmark data]", "15x throughput gain"])
        assert passed

    def test_what_if_hook_fails(self):
        passed, issues = self._check(hook_draft="What if you could share KV caches across requests?")
        assert not passed
        assert any("banned opener" in i.lower() or "what if" in i.lower() for i in issues)

    def test_introducing_hook_fails(self):
        passed, issues = self._check(hook_draft="Introducing LMCache, a new library for KV cache sharing.")
        assert not passed

    def test_low_confidence_fails(self):
        passed, issues = self._check(confidence=0.2)
        assert not passed
        assert any("confidence" in i.lower() for i in issues)

    def test_confidence_at_threshold_passes(self):
        passed, _ = self._check(confidence=0.3)
        assert passed

    def test_multiple_issues_all_reported(self):
        passed, issues = self._check(
            angle="makes it easier to work with AI",
            evidence=["[no benchmark data]"],
            hook_draft="What if you could improve your AI pipeline?",
            confidence=0.1,
        )
        assert not passed
        assert len(issues) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# 5. Research evaluation prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchEvalPrompt:
    def test_prompt_contains_all_three_dimensions(self):
        from backend.mcp.resources.prompts import research_evaluation_prompt
        prompt = research_evaluation_prompt(
            angle="LMCache eliminates recomputation",
            evidence=["15x throughput"],
            hook_draft="Your LLM burns 50% compute on repeated work.",
            confidence=0.9,
            content_type="github_tool",
        )
        assert "SPECIFICITY" in prompt
        assert "HOOK_VIABILITY" in prompt
        assert "EVIDENCE_GROUNDING" in prompt

    def test_prompt_has_threshold(self):
        from backend.mcp.resources.prompts import research_evaluation_prompt
        prompt = research_evaluation_prompt("angle", ["ev"], "hook", 0.8, "github_tool")
        assert "6.0" in prompt  # pass threshold

    def test_prompt_includes_brief_fields(self):
        from backend.mcp.resources.prompts import research_evaluation_prompt
        prompt = research_evaluation_prompt("my angle here", ["ev1"], "my hook draft", 0.75, "research_paper")
        assert "my angle here" in prompt
        assert "my hook draft" in prompt
        assert "research_paper" in prompt

    def test_prompt_requires_json_output(self):
        from backend.mcp.resources.prompts import research_evaluation_prompt
        prompt = research_evaluation_prompt("a", ["e"], "h", 0.5, "github_tool")
        assert "JSON" in prompt or "json" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 6. evaluate_research_brief: programmatic path (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateResearchBrief:
    def _brief(self, **overrides):
        defaults = dict(
            content_type="github_tool",
            angle="LMCache eliminates KV cache recomputation — saves 50% compute",
            evidence=["15x throughput gain", "3-10x TTFT reduction"],
            contrast="Before: cache dies per request. After: shared persistent layer.",
            limitations=["Requires vLLM or SGLang"],
            hook_recommendation="curiosity-gap",
            hook_draft="Your LLM inference is burning 50% of its compute on repeated work.",
            confidence=0.9,
            data_gaps=["no disk-offload latency data"],
        )
        defaults.update(overrides)
        return defaults

    @pytest.mark.asyncio
    async def test_programmatic_failure_skips_llm(self):
        """When programmatic guardrails fail, LLM judge is never called."""
        from backend.evaluation.evaluator import evaluate_research_brief
        bad_brief = self._brief(
            angle="makes it easier",
            evidence=["[no benchmark data]"],
            hook_draft="What if you could improve your pipeline?",
            confidence=0.1,
        )
        with patch("backend.evaluation.evaluator._ollama_chat") as mock_llm:
            result = await evaluate_research_brief(bad_brief)
        mock_llm.assert_not_called()
        assert not result["passed"]
        assert result["stage"] == "programmatic"
        assert len(result["issues"]) >= 1

    @pytest.mark.asyncio
    async def test_programmatic_pass_calls_llm_judge(self):
        """When programmatic passes, the LLM judge is called once."""
        from backend.evaluation.evaluator import evaluate_research_brief
        llm_response = json.dumps({
            "specificity": 8.0, "hook_viability": 7.5, "evidence_grounding": 9.0,
            "overall": 8.17, "passed": True, "issues": []
        })
        with patch("backend.evaluation.evaluator._ollama_chat", new_callable=AsyncMock, return_value=llm_response):
            result = await evaluate_research_brief(self._brief())
        assert result["passed"]
        assert result["stage"] == "llm"
        assert result["overall"] >= 6.0

    @pytest.mark.asyncio
    async def test_llm_judge_failure_falls_back_gracefully(self):
        """If LLM judge throws, the evaluation still returns passed=True (don't block pipeline)."""
        from backend.evaluation.evaluator import evaluate_research_brief
        with patch("backend.evaluation.evaluator._ollama_chat", new_callable=AsyncMock,
                   side_effect=Exception("LLM timeout")):
            result = await evaluate_research_brief(self._brief())
        assert result["passed"]  # programmatic passed, LLM failed → graceful fallback
        assert result["stage"] == "programmatic_fallback"

    @pytest.mark.asyncio
    async def test_llm_overall_recomputed_from_scores(self):
        """overall is recalculated from the 3 raw scores, not trusted from LLM output."""
        from backend.evaluation.evaluator import evaluate_research_brief
        # LLM claims overall=9.0 but individual scores average to 5.67
        llm_response = json.dumps({
            "specificity": 6.0, "hook_viability": 5.0, "evidence_grounding": 6.0,
            "overall": 9.0,  # dishonest/hallucinated — should be overridden
            "passed": True, "issues": []
        })
        with patch("backend.evaluation.evaluator._ollama_chat", new_callable=AsyncMock, return_value=llm_response):
            result = await evaluate_research_brief(self._brief())
        # Overall should be recomputed as (6+5+6)/3 = 5.67, not 9.0
        assert abs(result["overall"] - 5.67) < 0.1
        # 5.67 < 6.0 → should fail
        assert not result["passed"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pipeline edge: research_gate_decision
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchGateDecision:
    def _state(self, passed, attempts):
        return {
            "research_evaluation": {"passed": passed, "overall": 8.0 if passed else 4.0},
            "research_attempts": attempts,
        }

    def test_passed_evaluation_routes_to_pass(self):
        from backend.pipeline.edges import research_gate_decision
        assert research_gate_decision(self._state(passed=True, attempts=1)) == "pass"

    def test_failed_first_attempt_routes_to_retry(self):
        from backend.pipeline.edges import research_gate_decision
        assert research_gate_decision(self._state(passed=False, attempts=1)) == "retry"

    def test_failed_second_attempt_routes_to_fallback(self):
        from backend.pipeline.edges import research_gate_decision
        assert research_gate_decision(self._state(passed=False, attempts=2)) == "fallback"

    def test_failed_beyond_limit_routes_to_fallback(self):
        from backend.pipeline.edges import research_gate_decision
        assert research_gate_decision(self._state(passed=False, attempts=5)) == "fallback"

    def test_no_evaluation_defaults_to_retry(self):
        from backend.pipeline.edges import research_gate_decision
        state = {"research_evaluation": None, "research_attempts": 0}
        assert research_gate_decision(state) == "retry"

    def test_passed_after_retry_still_routes_pass(self):
        from backend.pipeline.edges import research_gate_decision
        assert research_gate_decision(self._state(passed=True, attempts=2)) == "pass"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pipeline nodes: research_content_node and evaluate_research_node
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchPipelineNodes:
    def _base_state(self, **overrides):
        state = {
            "current_item": {
                "title": "LMCache",
                "url": "https://github.com/LMCache/LMCache",
                "source": "github",
                "summary": "KV cache sharing library for vLLM",
            },
            "extracted_content": "LMCache provides 15x throughput gain and 3-10x TTFT reduction "
                                 "by making KV caches persistent and shareable across vLLM instances.",
            "research_brief": None,
            "research_attempts": 0,
            "research_evaluation": None,
            "research_failed": False,
            "logs": [],
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_research_node_stores_brief_on_success(self):
        from backend.pipeline.graph import research_content_node
        mock_result = {
            "success": True,
            "content_type": "github_tool",
            "angle": "LMCache eliminates KV cache recomputation",
            "evidence": ["15x throughput gain"],
            "contrast": "Before: cache dies. After: shared persistent.",
            "limitations": ["Requires vLLM"],
            "hook_recommendation": "curiosity-gap",
            "hook_draft": "Your LLM burns 50% compute on repeated work.",
            "confidence": 0.9,
            "data_gaps": [],
        }
        with patch("backend.pipeline.graph.llm_research_content", new_callable=AsyncMock,
                   return_value=mock_result):
            result = await research_content_node(self._base_state())
        assert result["research_brief"] == mock_result
        assert result["research_attempts"] == 1
        assert result["step"] == "research"

    @pytest.mark.asyncio
    async def test_research_node_clears_brief_on_failure(self):
        from backend.pipeline.graph import research_content_node
        mock_result = {"success": False, "error": "LLM timeout", "angle": "", "confidence": 0.0,
                       "content_type": "github_tool", "evidence": [], "contrast": "[not stated]",
                       "limitations": ["[not mentioned]"], "hook_recommendation": "bold-statement",
                       "hook_draft": "", "data_gaps": []}
        with patch("backend.pipeline.graph.llm_research_content", new_callable=AsyncMock,
                   return_value=mock_result):
            result = await research_content_node(self._base_state())
        assert result["research_brief"] is None  # success=False → no brief
        assert result["research_attempts"] == 1

    @pytest.mark.asyncio
    async def test_research_node_escalates_temperature_on_retry(self):
        """Second attempt uses higher temperature than first."""
        from backend.pipeline.graph import research_content_node
        temperatures_used = []
        async def mock_research(title, content, source, source_url, previous_issues, temperature):
            temperatures_used.append(temperature)
            return {"success": True, "content_type": "github_tool", "angle": "x",
                    "evidence": ["data"], "contrast": "c", "limitations": [],
                    "hook_recommendation": "bold-statement", "hook_draft": "x", "confidence": 0.5, "data_gaps": []}

        with patch("backend.pipeline.graph.llm_research_content", side_effect=mock_research):
            # First attempt
            await research_content_node(self._base_state(research_attempts=0))
            # Second attempt (retry)
            await research_content_node(self._base_state(research_attempts=1))

        assert temperatures_used[0] < temperatures_used[1], (
            f"Retry should use higher temp: {temperatures_used}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_research_node_passed_brief(self):
        from backend.pipeline.graph import evaluate_research_node
        good_brief = {
            "success": True, "content_type": "github_tool",
            "angle": "LMCache eliminates KV cache recomputation — saves 50% compute",
            "evidence": ["15x throughput gain", "3-10x TTFT reduction"],
            "contrast": "Before: cache dies. After: shared persistent layer.",
            "limitations": ["Requires vLLM"], "hook_recommendation": "curiosity-gap",
            "hook_draft": "Your LLM burns 50% of its compute on repeated work.",
            "confidence": 0.9, "data_gaps": [],
        }
        llm_response = json.dumps({
            "specificity": 8.5, "hook_viability": 8.0, "evidence_grounding": 9.0,
            "overall": 8.5, "passed": True, "issues": []
        })
        with patch("backend.evaluation.evaluator._ollama_chat", new_callable=AsyncMock,
                   return_value=llm_response):
            result = await evaluate_research_node(
                self._base_state(research_brief=good_brief, research_attempts=1)
            )
        assert result["research_evaluation"]["passed"]
        assert not result.get("research_failed", False)
        assert result["research_brief"] is not None  # brief preserved

    @pytest.mark.asyncio
    async def test_evaluate_research_node_no_brief_goes_fallback(self):
        from backend.pipeline.graph import evaluate_research_node
        result = await evaluate_research_node(
            self._base_state(research_brief=None, research_attempts=1)
        )
        assert result["research_failed"]
        assert not result["research_evaluation"]["passed"]

    @pytest.mark.asyncio
    async def test_evaluate_research_node_exhausted_retries_sets_failed(self):
        from backend.pipeline.graph import evaluate_research_node
        weak_brief = {
            "success": True, "content_type": "github_tool",
            "angle": "makes it easier",  # generic — will fail programmatic
            "evidence": ["[no benchmark data]"],
            "contrast": "[not stated]", "limitations": [], "hook_recommendation": "bold-statement",
            "hook_draft": "What if you could improve your pipeline?",  # banned opener
            "confidence": 0.1, "data_gaps": [],
        }
        result = await evaluate_research_node(
            self._base_state(research_brief=weak_brief, research_attempts=2)  # exhausted
        )
        assert result["research_failed"]
        assert result["research_brief"] is None  # cleared in fallback

    @pytest.mark.asyncio
    async def test_generate_post_receives_brief(self):
        """generate_post_node passes research_brief to llm_generate_post when available."""
        from backend.pipeline.graph import generate_post_node
        captured = {}
        async def mock_generate(**kwargs):
            captured.update(kwargs)
            return {
                "post_text": "Test post content. https://example.com #AI #ML #LLM #Agents",
                "hook": "Test hook", "body": "Test body", "takeaway": "Test takeaway",
                "hashtags": ["AI", "ML", "LLM", "Agents"], "url": "https://example.com",
                "char_count": 60, "hook_pattern_used": "curiosity-gap",
                "seo_keywords_used": [], "emoji_placement": "",
            }

        good_brief = {
            "success": True, "content_type": "github_tool",
            "angle": "LMCache eliminates KV cache recomputation",
            "evidence": ["15x throughput gain"], "contrast": "Before/after contrast",
            "limitations": [], "hook_recommendation": "curiosity-gap",
            "hook_draft": "Your LLM burns 50% compute.", "confidence": 0.9, "data_gaps": [],
        }

        state = {
            "current_item": {"title": "LMCache", "url": "https://github.com/LMCache/LMCache",
                             "source": "github", "summary": "KV cache library"},
            "extracted_content": "LMCache provides 15x throughput...",
            "research_brief": good_brief,
            "research_failed": False,
            "generation_attempts": 0,
            "current_evaluation": None,
            "logs": [],
            "run_config": {},
            "trending_topics": [],
        }

        with patch("backend.pipeline.graph.llm_generate_post", side_effect=mock_generate), \
             patch("backend.memory.hindsight_client.get_style_guidance", new_callable=AsyncMock, return_value=""), \
             patch("backend.mcp.resources.prompts.lessons_learned_prompt", return_value=""):
            await generate_post_node(state)

        assert "research_brief" in captured
        assert captured["research_brief"] == good_brief

    @pytest.mark.asyncio
    async def test_generate_post_no_brief_when_research_failed(self):
        """When research_failed=True, research_brief is NOT passed to generate_post."""
        from backend.pipeline.graph import generate_post_node
        captured = {}
        async def mock_generate(**kwargs):
            captured.update(kwargs)
            return {
                "post_text": "Test post. https://example.com #AI #ML #LLM #Agents",
                "hook": "h", "body": "b", "takeaway": "t",
                "hashtags": ["AI", "ML", "LLM", "Agents"], "url": "https://example.com",
                "char_count": 50, "hook_pattern_used": "bold-statement",
                "seo_keywords_used": [], "emoji_placement": "",
            }

        state = {
            "current_item": {"title": "X", "url": "https://x.com", "source": "github", "summary": "y"},
            "extracted_content": "content",
            "research_brief": {"angle": "some brief", "confidence": 0.8},
            "research_failed": True,  # failed → brief should NOT be passed
            "generation_attempts": 0,
            "current_evaluation": None,
            "logs": [],
            "run_config": {},
            "trending_topics": [],
        }

        with patch("backend.pipeline.graph.llm_generate_post", side_effect=mock_generate), \
             patch("backend.memory.hindsight_client.get_style_guidance", new_callable=AsyncMock, return_value=""), \
             patch("backend.mcp.resources.prompts.lessons_learned_prompt", return_value=""):
            await generate_post_node(state)

        assert captured.get("research_brief") is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Pipeline state: research fields reset between items
# ─────────────────────────────────────────────────────────────────────────────

class TestPickNextItemResetsResearch:
    @pytest.mark.asyncio
    async def test_research_fields_reset_on_next_item(self):
        from backend.pipeline.graph import pick_next_item_node
        state = {
            "filtered_items": [
                {"title": "Item1", "url": "https://a.com", "source": "github", "summary": ""},
                {"title": "Item2", "url": "https://b.com", "source": "github", "summary": ""},
            ],
            "current_item_index": 0,
            "run_config": {"max_posts": 5},
            # Leftover research state from previous item
            "research_brief": {"angle": "leftover brief"},
            "research_attempts": 2,
            "research_evaluation": {"passed": True},
            "research_failed": True,
            "logs": [],
        }
        result = await pick_next_item_node(state)
        assert result["research_brief"] is None
        assert result["research_attempts"] == 0
        assert result["research_evaluation"] is None
        assert result["research_failed"] == False


# ─────────────────────────────────────────────────────────────────────────────
# 10. Graph topology: new nodes wired correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphTopology:
    def test_all_12_nodes_present(self):
        from backend.pipeline.graph import build_pipeline
        g = build_pipeline()
        node_names = set(g.nodes.keys()) - {"__start__"}
        expected = {
            "fetch_content", "filter_and_score", "pick_next_item",
            "extract_article", "research_content", "evaluate_research",
            "generate_post", "evaluate_post", "check_duplicate",
            "save_post", "reject_post", "finalize",
        }
        assert expected == node_names, f"Node mismatch: {node_names ^ expected}"

    def test_graph_compiles_without_error(self):
        from backend.pipeline.graph import build_pipeline
        g = build_pipeline()
        assert g is not None


# ─────────────────────────────────────────────────────────────────────────────
# 11. generate_post brief injection: user prompt content
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratePostBriefInjection:
    """Tests that generate_post builds different user prompts with/without a brief."""

    @pytest.mark.asyncio
    async def test_brief_context_in_user_prompt_when_provided(self):
        """When a research brief is passed, user prompt contains RESEARCH BRIEF section."""
        from backend.mcp.tools import llm as llm_mod
        captured_messages = []

        async def mock_generate_with_model(messages, response_model, temperature, seed=None):
            captured_messages.extend(messages)
            from backend.models import LinkedInPost
            return LinkedInPost(
                angle="LMCache eliminates KV cache recomputation",
                hook="Your LLM burns 50% compute on repeated work.",
                body="KV cache is recomputed per request in standard vLLM setups.\n\nLMCache offloads it to a shared layer.",
                takeaway="Stop letting your GPUs do the same homework twice.",
                hashtags=["MachineLearning", "AIInfrastructure", "LLM", "MLOps"],
                url="https://github.com/LMCache/LMCache",
                hook_pattern_used="curiosity-gap",
            )

        brief = {
            "content_type": "github_tool",
            "angle": "LMCache eliminates KV cache recomputation",
            "evidence": ["15x throughput gain"],
            "contrast": "Before: cache dies. After: persistent shared layer.",
            "limitations": ["Requires vLLM"],
            "hook_recommendation": "curiosity-gap",
            "hook_draft": "Your LLM burns 50% compute.",
            "confidence": 0.9,
            "data_gaps": [],
        }

        with patch.object(llm_mod, "_generate_with_model", side_effect=mock_generate_with_model):
            await llm_mod.generate_post(
                title="LMCache", summary="KV cache library", source="github",
                source_url="https://github.com/LMCache/LMCache",
                content="KV cache content here", research_brief=brief,
            )

        user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
        assert "RESEARCH BRIEF" in user_msg
        assert "LMCache eliminates KV cache recomputation" in user_msg
        assert "15x throughput gain" in user_msg

    @pytest.mark.asyncio
    async def test_no_brief_uses_standard_prompt(self):
        """Without a brief, standard prompt is used (no RESEARCH BRIEF section)."""
        from backend.mcp.tools import llm as llm_mod
        captured_messages = []

        async def mock_generate_with_model(messages, response_model, temperature, seed=None):
            captured_messages.extend(messages)
            from backend.models import LinkedInPost
            return LinkedInPost(
                angle="angle", hook="hook", body="body", takeaway="takeaway",
                hashtags=["AI", "ML", "LLM", "Agents"],
                url="https://example.com", hook_pattern_used="bold-statement",
            )

        with patch.object(llm_mod, "_generate_with_model", side_effect=mock_generate_with_model):
            await llm_mod.generate_post(
                title="X", summary="summary", source="github",
                source_url="https://example.com", research_brief=None,
            )

        user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
        assert "RESEARCH BRIEF" not in user_msg

    @pytest.mark.asyncio
    async def test_low_confidence_brief_not_injected(self):
        """Brief with confidence < 0.3 is treated as no brief."""
        from backend.mcp.tools import llm as llm_mod
        captured_messages = []

        async def mock_generate_with_model(messages, response_model, temperature, seed=None):
            captured_messages.extend(messages)
            from backend.models import LinkedInPost
            return LinkedInPost(
                angle="a", hook="h", body="b", takeaway="t",
                hashtags=["AI", "ML", "LLM", "Agents"],
                url="https://example.com", hook_pattern_used="bold-statement",
            )

        low_conf_brief = {"angle": "some angle", "confidence": 0.1, "evidence": [], "data_gaps": []}

        with patch.object(llm_mod, "_generate_with_model", side_effect=mock_generate_with_model):
            await llm_mod.generate_post(
                title="X", summary="summary", source="github",
                source_url="https://example.com", research_brief=low_conf_brief,
            )

        user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
        assert "RESEARCH BRIEF" not in user_msg
