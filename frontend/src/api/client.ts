import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

/* ---------- Types ---------- */

export interface Post {
  id: string;
  title: string;
  post_text: string;
  source: string;
  original_url: string;
  image_url: string | null;
  quality_score: number;
  relevance_score: number;
  faithfulness_score: number;
  trending_boost: number;
  final_score: number;
  char_count: number;
  hashtags: string[] | null;
  is_used: boolean;
  created_at: string;
  generation_attempts: number;
  pipeline_run_id: string | null;
  human_rating_avg?: number;
  human_feedback_count?: number;
  detailed_eval_id?: string | null;
}

export interface PaginatedPosts {
  posts: Post[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

export interface PostsParams {
  page?: number;
  per_page?: number;
  source?: string;
  search?: string;
  sort_by?: string;
  min_score?: number;
  is_used?: boolean;
}

export interface SourceDistEntry {
  source: string;
  total: number;
  accepted: number;
  avg_quality: number;
  avg_score: number;
}

export interface PostStats {
  total_posts: number;
  used_posts: number;
  avg_relevance: number;
  avg_quality: number;
  avg_score: number;
  avg_length: number;
  source_distribution: SourceDistEntry[];
}

export interface Source {
  id: string;
  source_type: string;
  name: string;
  url: string;
  category: string;
  params: Record<string, unknown>;
  is_active: boolean;
  priority: number;
  total_fetched: number;
  total_accepted: number;
  avg_quality: number;
  last_fetched_at: string | null;
  last_error?: string | null;
  last_error_at?: string | null;
  consecutive_failures?: number;
  health?: "healthy" | "degraded" | "critical" | "disabled";
}

export interface SourceCreate {
  source_type: string;
  name: string;
  url: string;
  category?: string;
  is_active?: boolean;
  priority?: number;
}

export interface SourceHealthSummary {
  total: number;
  healthy: number;
  unhealthy: number;
  disabled: number;
}

export interface ConfigEntry {
  key: string;
  value: unknown;
  description?: string;
  updated_at?: string;
}

export interface PipelineRunConfig {
  max_posts?: number;
  max_age_days?: number;
  max_retries?: number;
  limit_per_source?: number;
  sources?: string[];
}

export interface PipelineRun {
  run_id: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  items_fetched: number;
  items_filtered: number;
  posts_accepted: number;
  posts_rejected: number;
  avg_quality: number;
  pass_rate: number;
}

export interface McpTool {
  name: string;
  description: string;
}

/* ---------- Detailed Evaluation (v2) ---------- */

export interface DetailedEvaluation {
  id: string;
  post_id: string;
  evaluated_at: string;
  pipeline_run_id: string;
  answer_relevancy: number;
  faithfulness: number;
  hallucination: number;
  bias: number;
  toxicity: number;
  linkedin_quality: number;
  overall_score: number;
  passed: boolean;
  metric_details: Record<string, unknown>;
  issues: string[];
  strengths: string[];
  judge_model: string;
  evaluation_version: string;
}

/* ---------- Human Feedback ---------- */

export interface HumanFeedback {
  id: string;
  post_id: string;
  created_at: string;
  rating: number;
  text_feedback: string;
  is_good_example: boolean;
  feedback_weight: number;
  feedback_source: string;
}

export interface HumanFeedbackCreate {
  rating: number;
  text_feedback?: string;
  is_good_example?: boolean;
}

export interface FeedbackSummary {
  overall: {
    avg_rating: number;
    total_feedbacks: number;
    unique_posts_rated: number;
  };
  by_source: { source: string; avg_rating: number; count: number }[];
  top_rated_posts: { id: string; title: string; avg_rating: number; feedback_count: number }[];
}

/* ---------- Posts ---------- */

export async function getPosts(params: PostsParams = {}): Promise<PaginatedPosts> {
  const { data } = await api.get("/posts", { params });
  return data;
}

export async function getPostStats(): Promise<PostStats> {
  const { data } = await api.get("/posts/stats/summary");
  return data;
}

export async function toggleUsed(id: string): Promise<{ id: string; is_used: boolean }> {
  const { data } = await api.patch(`/posts/${id}/used`);
  return data;
}

export async function deletePost(id: string): Promise<void> {
  await api.delete(`/posts/${id}`);
}

/* ---------- Sources ---------- */

export async function getSources(): Promise<Source[]> {
  const { data } = await api.get("/sources");
  return data.sources;
}

export async function getSourceHealth(): Promise<{ sources: Source[]; summary: SourceHealthSummary }> {
  const { data } = await api.get("/sources/health");
  return data;
}

export async function createSource(source: SourceCreate): Promise<{ id: string; created: boolean }> {
  const { data } = await api.post("/sources", source);
  return data;
}

export async function updateSource(
  id: string,
  updates: Partial<SourceCreate & { is_active: boolean; priority: number }>,
): Promise<{ updated: boolean }> {
  const { data } = await api.patch(`/sources/${id}`, updates);
  return data;
}

export async function deleteSource(id: string): Promise<void> {
  await api.delete(`/sources/${id}`);
}

/* ---------- Config ---------- */

export async function getConfig(): Promise<ConfigEntry[]> {
  const { data } = await api.get("/config");
  return data.configs;
}

export async function setConfig(key: string, value: unknown): Promise<{ key: string; updated: boolean }> {
  const { data } = await api.put(`/config/${key}`, { value });
  return data;
}

/* ---------- Pipeline / Agent ---------- */

export async function startPipeline(config: PipelineRunConfig): Promise<{ status: string; message: string }> {
  const { data } = await api.post("/agent/run", config);
  return data;
}

export async function getPipelineRuns(): Promise<PipelineRun[]> {
  const { data } = await api.get("/agent/runs");
  return data.runs;
}

export async function getRunningPipeline(): Promise<{ running: boolean; run_id: string | null }> {
  const { data } = await api.get("/agent/running");
  return data;
}

export async function cancelPipeline(): Promise<{ cancelled: boolean }> {
  const { data } = await api.post("/agent/cancel");
  return data;
}

/* ---------- MCP ---------- */

export async function getMcpTools(): Promise<McpTool[]> {
  const { data } = await api.get("/mcp/tools");
  return data.tools;
}

export interface MetricAverages {
  total: number;
  overall: number;
  answer_relevancy: number;
  faithfulness: number;
  hallucination: number;
  bias: number;
  toxicity: number;
  linkedin_quality: number;
}

export interface PerRunMetrics extends MetricAverages {
  pipeline_run_id: string;
  started_at: string;
  eval_count: number;
}

export interface MetricsSummary {
  overall: MetricAverages;
  per_run: PerRunMetrics[];
}

/* ---------- Evaluations (v2) ---------- */

export async function getMetricsSummary(): Promise<MetricsSummary> {
  const { data } = await api.get("/evaluations/metrics/summary");
  return data;
}

export interface EvaluationThresholds {
  thresholds: Record<string, number>;
  weights: Record<string, number>;
  overall_pass_threshold: number;
}

export async function getEvaluationThresholds(): Promise<EvaluationThresholds> {
  const { data } = await api.get("/evaluations/thresholds");
  return data;
}

export async function getPostEvaluation(postId: string): Promise<DetailedEvaluation | null> {
  try {
    const { data } = await api.get(`/evaluations/post/${postId}`);
    return data.found ? data.evaluation : null;
  } catch {
    return null;
  }
}

/* ---------- Human Feedback ---------- */

export async function submitFeedback(postId: string, feedback: HumanFeedbackCreate): Promise<{ id: string }> {
  const { data } = await api.post(`/feedback/${postId}`, feedback);
  return data;
}

export async function getPostFeedback(postId: string): Promise<HumanFeedback[]> {
  const { data } = await api.get(`/feedback/${postId}`);
  return data.feedbacks;
}

export async function getFeedbackSummary(): Promise<FeedbackSummary> {
  const { data } = await api.get("/feedback/summary/stats");
  return data;
}

export async function getGoodExamples(): Promise<{ examples: Post[]; count: number }> {
  const { data } = await api.get("/feedback/good-examples/list");
  return data;
}

export default api;
