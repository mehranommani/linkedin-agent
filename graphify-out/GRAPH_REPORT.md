# Graph Report - backend  (2026-04-09)

## Corpus Check
- 50 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 611 nodes · 1107 edges · 26 communities detected
- Extraction: 56% EXTRACTED · 44% INFERRED · 0% AMBIGUOUS · INFERRED: 488 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `ConfigManager` - 128 edges
2. `ContentItem` - 67 edges
3. `BaseSource` - 55 edges
4. `LinkedInPost` - 25 edges
5. `PipelineState` - 24 edges
6. `ResearchBrief` - 23 edges
7. `GitHubSource` - 17 edges
8. `execute()` - 15 edges
9. `get()` - 14 edges
10. `_log()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `LinkedInPostSignature (DSPy)` --semantically_similar_to--> `LinkedInPost`  [INFERRED] [semantically similar]
  backend/dspy_modules/post_generator.py → /Users/mehran/Documents/GitHub/linkedin-agent/backend/models.py
- `DuckDB` --rationale_for--> `_persist_evaluation()`  [INFERRED]
  backend/requirements.txt → /Users/mehran/Documents/GitHub/linkedin-agent/backend/evaluation/evaluator.py
- `check_duplicate (MCP tool)` --calls--> `ConfigManager`  [EXTRACTED]
  backend/mcp/tools/evaluator.py → /Users/mehran/Documents/GitHub/linkedin-agent/backend/config.py
- `fetch_source (MCP tool)` --shares_data_with--> `ContentItem`  [INFERRED]
  backend/mcp/tools/sources.py → /Users/mehran/Documents/GitHub/linkedin-agent/backend/models.py
- `insert_post (MCP tool)` --shares_data_with--> `PostCreate`  [INFERRED]
  backend/mcp/tools/database.py → /Users/mehran/Documents/GitHub/linkedin-agent/backend/models.py

## Hyperedges (group relationships)
- **LangGraph Pipeline Nodes** — pipeline_graph_fetchcontent, pipeline_graph_filterandscore, pipeline_graph_picknextitem, pipeline_edges_qualitygate, pipeline_edges_shouldcontinue, pipeline_edges_researchgate [EXTRACTED 1.00]
- **MCP Tool Suite** — mcp_tools_db_insertpost, mcp_tools_llm_generatepost, mcp_tools_sources_fetchsource, mcp_tools_eval_rundetailed, mcp_resources_prompts_postgen [EXTRACTED 1.00]
- **Hindsight Memory Layer** — hindsight_client_storegenerationexp, hindsight_client_storefeedback, hindsight_client_storefailedexp, hindsight_client_getstyleguidance, hindsight_client_storestylex [EXTRACTED 0.95]
- **All Content Sources Implement BaseSource** — devto_DevToSource, hackernews_HackerNewsSource, rss_RSSSource, papers_PapersWithCodeSource, producthunt_ProductHuntSource, github_GitHubSource, reddit_RedditSource, arxiv_ArxivSource [EXTRACTED 1.00]
- **CRUD API Routers** — posts_PostsAPI, sources_SourcesAPI, feedback_FeedbackAPI, config_api_ConfigAPI [INFERRED 0.85]
- **Parallel LLM Metric Evaluators** — metrics_evaluate_relevancy, metrics_evaluate_faithfulness_hallucination, metrics_evaluate_bias_toxicity, metrics_evaluate_linkedin_quality [EXTRACTED 1.00]

## Communities

### Community 0 - "ArXiv Content Sources"
Cohesion: 0.05
Nodes (81): ABC, ArxivSource, ArxivSource, _parse_datetime(), ArXiv Source ============ Fetches recent papers from the ArXiv API filtered by c, Parse an ISO-8601 timestamp from ArXiv., Recent papers from ArXiv, filtered by category and age., Fetch papers from ArXiv.          ``params`` keys         --------------- (+73 more)

### Community 1 - "Config API Layer"
Cohesion: 0.03
Nodes (86): list_config(), Config API ========== Dynamic config CRUD., List all config entries., ConfigManager, Reads/writes config from the DuckDB config table., get_common_issues(), get_metrics_summary(), get_post_evaluations() (+78 more)

### Community 2 - "DSPy Post Generation"
Cohesion: 0.08
Nodes (54): get_post_generator (DSPy Refine), LinkedInPostSignature (DSPy), post_quality_reward, apply_linkedin_bold, batch_judge_relevance(), _chat_with_fallback(), close_client(), _cloud_chat() (+46 more)

### Community 3 - "Pipeline Edge Routing"
Cohesion: 0.06
Nodes (52): quality_gate_decision(), Pipeline Edge Functions ====================== Conditional routing functions for, Decide what to do after evaluation: accept, retry, or reject.      Uses the v2 s, Decide whether to process more items or finish., Decide what to do after research evaluation: pass, retry, or fallback.      - pa, research_gate_decision(), should_continue(), _after_dedup_decision() (+44 more)

### Community 4 - "Feedback & Examples"
Cohesion: 0.05
Nodes (47): BaseModel, feedback_summary(), get_good_examples(), get_post_feedback(), Feedback API ============ CRUD for human feedback on generated posts. Supports r, Aggregate feedback stats: avg rating by source, most-rated posts., Get posts marked as good examples (for few-shot prompting)., Submit human feedback for a post and store in Hindsight for autonomous learning. (+39 more)

### Community 5 - "SQLite Database Layer"
Cohesion: 0.06
Nodes (43): check_url_exists(), close(), execute(), fetch_all(), fetch_df(), fetch_one(), get_active_sources(), get_config() (+35 more)

### Community 6 - "Hindsight Memory Client"
Cohesion: 0.08
Nodes (43): check_health(), close_client(), ensure_bank(), _get_bank_id(), _get_base_url(), get_client(), _get_config(), get_style_guidance() (+35 more)

### Community 7 - "MCP Tool Handlers"
Cohesion: 0.07
Nodes (35): fetch_all (SQL helper), fetch_one (SQL helper), lessons_learned_prompt, check_url_exists (MCP tool), get_active_sources (MCP tool), get_recent_issues (MCP tool), query_posts (MCP tool), check_duplicate (MCP tool) (+27 more)

### Community 8 - "FastAPI App Core"
Cohesion: 0.1
Nodes (22): get_connection (SQLite singleton), FastAPI App (main), _configure_dspy(), health_check(), lifespan(), list_mcp_tools(), LinkedIn AI Content Agent - Backend =================================== FastAPI, Startup and shutdown events. (+14 more)

### Community 9 - "API Router Hub"
Cohesion: 0.12
Nodes (18): Agent Pipeline API Router, Config API Router, Evaluations API Router, Evaluation Orchestrator, Feedback API Router, Six-Metric Evaluation Engine, Posts API Router, FastAPI (+10 more)

### Community 10 - "Config & Banned Phrases"
Cohesion: 0.2
Nodes (15): banned_phrases(), emoji_style(), evaluation(), evaluation_v2(), feedback_config(), generation(), get(), hindsight() (+7 more)

### Community 11 - "Agent Pipeline Control"
Cohesion: 0.16
Nodes (17): cancel_pipeline(), cleanup_stale_runs(), get_pipeline_status(), get_running_pipeline(), list_runs(), _parse_json_list(), Agent API ========= Start/stop/status for pipeline runs., Return the currently running pipeline run_id, or null if idle.      Checks both (+9 more)

### Community 12 - "Posts CRUD"
Cohesion: 0.18
Nodes (9): get_post(), list_posts(), post_stats(), Posts API ========= CRUD + paginated listing + stats for LinkedIn posts., Paginated post listing with filters., Dashboard summary statistics., Get a single post by ID., Toggle the used status of a post. (+1 more)

### Community 13 - "Post Generator Module"
Cohesion: 0.22
Nodes (9): _build_post_generator(), get_post_generator(), LinkedInPostSignature, post_quality_reward(), DSPy Post Generator =================== Defines the LinkedInPostSignature and wr, Return (and lazily create) the module-level post generator singleton., Generate a LinkedIn post about a software project for a professional audience., Returns 1.0 if post passes all programmatic guardrails, 0.0 otherwise. (+1 more)

### Community 14 - "LinkedIn Formatter"
Cohesion: 0.4
Nodes (5): apply_linkedin_bold(), LinkedIn Unicode Bold Formatter ================================ LinkedIn doesn', Convert ASCII letters and digits in *s* to Unicode bold equivalents., Apply Unicode bold to:       1. The first (hook) line — entirely bolded       2., to_bold()

### Community 15 - "MCP Composite Server"
Cohesion: 0.4
Nodes (5): get_server(), MCP Composite Server ==================== Single MCP server exposing all tools a, Import and register all tool modules.      Each module uses the @mcp_server.tool, Get the MCP server with all tools registered., register_all_tools()

### Community 16 - "API Router"
Cohesion: 1.0
Nodes (1): API Router ========== Aggregates all API routers.

### Community 17 - "Config Cache Read"
Cohesion: 1.0
Nodes (1): Get a config value. Caches after first read.

### Community 18 - "Config Write"
Cohesion: 1.0
Nodes (1): Set a config value in the database.

### Community 19 - "Config Cache Clear"
Cohesion: 1.0
Nodes (1): Clear cache. If key given, clear only that key.

### Community 20 - "Post Assembly"
Cohesion: 1.0
Nodes (1): Assemble the complete LinkedIn post.

### Community 21 - "Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Article Extractor Tool"
Cohesion: 1.0
Nodes (1): extract_article (MCP tool)

### Community 23 - "Pydantic"
Cohesion: 1.0
Nodes (1): Pydantic

### Community 24 - "Feedparser"
Cohesion: 1.0
Nodes (1): Feedparser

### Community 25 - "BeautifulSoup4"
Cohesion: 1.0
Nodes (1): BeautifulSoup4

## Knowledge Gaps
- **123 isolated node(s):** `ConfigManager ============= Reads configuration from DuckDB config table. Provid`, `Reads/writes config from the DuckDB config table.`, `Get a config value. Caches after first read.`, `Set a config value in the database.`, `Clear cache. If key given, clear only that key.` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `API Router`** (2 nodes): `router.py`, `API Router ========== Aggregates all API routers.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Cache Read`** (1 nodes): `Get a config value. Caches after first read.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Write`** (1 nodes): `Set a config value in the database.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Cache Clear`** (1 nodes): `Clear cache. If key given, clear only that key.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Post Assembly`** (1 nodes): `Assemble the complete LinkedIn post.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Article Extractor Tool`** (1 nodes): `extract_article (MCP tool)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pydantic`** (1 nodes): `Pydantic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Feedparser`** (1 nodes): `Feedparser`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BeautifulSoup4`** (1 nodes): `BeautifulSoup4`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConfigManager` connect `Config API Layer` to `ArXiv Content Sources`, `DSPy Post Generation`, `Pipeline Edge Routing`, `SQLite Database Layer`, `Hindsight Memory Client`, `MCP Tool Handlers`, `Config & Banned Phrases`?**
  _High betweenness centrality (0.609) - this node is a cross-community bridge._
- **Why does `ContentItem` connect `ArXiv Content Sources` to `Feedback & Examples`, `SQLite Database Layer`?**
  _High betweenness centrality (0.162) - this node is a cross-community bridge._
- **Why does `execute()` connect `SQLite Database Layer` to `Config API Layer`, `MCP Tool Handlers`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 117 inferred relationships involving `ConfigManager` (e.g. with `LangGraph Pipeline ================== Autonomous pipeline that fetches, filters,` and `Append a log message, broadcast via WebSocket if available, and return updated l`) actually correct?**
  _`ConfigManager` has 117 INFERRED edges - model-reasoned connections that need verification._
- **Are the 56 inferred relationships involving `ContentItem` (e.g. with `DevToSource` and `Dev.to Source ============= Fetches recent articles from the Dev.to public API,`) actually correct?**
  _`ContentItem` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 52 inferred relationships involving `BaseSource` (e.g. with `ContentItem` and `DevToSource`) actually correct?**
  _`BaseSource` has 52 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `LinkedInPost` (e.g. with `MCP LLM Tools ============= Tools for LLM operations: generate posts, batch rele` and `Get or create a persistent httpx client with connection pooling.`) actually correct?**
  _`LinkedInPost` has 21 INFERRED edges - model-reasoned connections that need verification._