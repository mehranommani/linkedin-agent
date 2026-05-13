import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import { getPipelineRuns, getMetricsSummary, getEvaluationThresholds, type PipelineRun } from "../api/client";
import PipelineStatus from "../components/PipelineStatus";
import { MetricBadge, METRIC_LABELS, THRESHOLDS } from "../components/EvaluationReport";

const METRIC_COLORS: Record<string, string> = {
  answer_relevancy: "#3b82f6",
  faithfulness: "#8b5cf6",
  hallucination: "#06b6d4",
  bias: "#f59e0b",
  toxicity: "#ef4444",
  linkedin_quality: "#22c55e",
};

export default function Evaluations() {
  const { data: runs, isLoading } = useQuery({
    queryKey: ["pipelineRuns"],
    queryFn: getPipelineRuns,
  });
  const { data: metricsSummary } = useQuery({
    queryKey: ["metricsSummary"],
    queryFn: getMetricsSummary,
  });
  const { data: thresholdsData } = useQuery({
    queryKey: ["evaluationThresholds"],
    queryFn: getEvaluationThresholds,
    staleTime: 5 * 60 * 1000,
  });
  const liveThresholds = thresholdsData?.thresholds ?? THRESHOLDS;

  const sortedRuns = (runs ?? []).slice().sort(
    (a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime(),
  );

  // Trend data for charts
  const trendData = sortedRuns.map((run, idx) => ({
    run: idx + 1,
    label: new Date(run.started_at).toLocaleDateString(),
    passRate:
      (run.posts_accepted + run.posts_rejected) > 0
        ? Math.round((run.posts_accepted / (run.posts_accepted + run.posts_rejected)) * 100)
        : 0,
    avgQuality: run.avg_quality ?? 0,
  }));

  // Per-metric trend from detailed_evaluations
  const perRunMetricData = (metricsSummary?.per_run ?? []).map((r, idx) => ({
    run: idx + 1,
    label: r.started_at ? new Date(r.started_at).toLocaleDateString() : `Run ${idx + 1}`,
    answer_relevancy: r.answer_relevancy,
    faithfulness: r.faithfulness,
    hallucination: r.hallucination,
    bias: r.bias,
    toxicity: r.toxicity,
    linkedin_quality: r.linkedin_quality,
  }));

  const overallMetrics = metricsSummary?.overall;
  const radarData = overallMetrics
    ? Object.keys(METRIC_LABELS).map((m) => ({
        metric: METRIC_LABELS[m],
        score: (overallMetrics as unknown as Record<string, number>)[m] ?? 0,
        threshold: THRESHOLDS[m] ?? 7.0,
        fullMark: 10,
      }))
    : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Evaluations</h1>

      {/* Overall Metric Averages */}
      {overallMetrics && overallMetrics.total > 0 && (
        <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Overall Metric Averages</h2>
            <span className="text-sm text-gray-400">{overallMetrics.total} posts evaluated</span>
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Radar overview */}
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 10]} tick={{ fill: "#6b7280", fontSize: 9 }} />
                <Radar name="Threshold" dataKey="threshold" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.08} strokeDasharray="4 4" />
                <Radar name="Avg Score" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
              </RadarChart>
            </ResponsiveContainer>
            {/* Metric badges */}
            <div className="grid grid-cols-2 gap-3 content-start">
              {Object.keys(METRIC_LABELS).map((m) => (
                <MetricBadge
                  key={m}
                  name={m}
                  score={(overallMetrics as unknown as Record<string, number>)[m] ?? 0}
                  thresholds={liveThresholds}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Per-metric Trend */}
      {perRunMetricData.length > 1 && (
        <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
          <h2 className="mb-4 text-lg font-semibold text-white">Per-Metric Trends</h2>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={perRunMetricData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="label" stroke="#9ca3af" tick={{ fontSize: 11 }} />
              <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} domain={[0, 10]} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: "0.5rem", color: "#f3f4f6" }}
                formatter={(value: number, name: string) => [value.toFixed(1), METRIC_LABELS[name] ?? name]}
              />
              {Object.keys(METRIC_LABELS).map((m) => (
                <Line
                  key={m}
                  type="monotone"
                  dataKey={m}
                  stroke={METRIC_COLORS[m] ?? "#6b7280"}
                  strokeWidth={1.5}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Pass Rate Trend */}
        <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
          <h2 className="mb-4 text-lg font-semibold text-white">
            Pass Rate Trend
          </h2>
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="label"
                  stroke="#9ca3af"
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  stroke="#9ca3af"
                  tick={{ fontSize: 11 }}
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1f2937",
                    border: "1px solid #374151",
                    borderRadius: "0.5rem",
                    color: "#f3f4f6",
                  }}
                  formatter={(value: number) => [`${value}%`, "Pass Rate"]}
                />
                <Line
                  type="monotone"
                  dataKey="passRate"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 4, fill: "#3b82f6" }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="py-16 text-center text-gray-500">No data yet</p>
          )}
        </div>

        {/* Average Quality Trend */}
        <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
          <h2 className="mb-4 text-lg font-semibold text-white">
            Average Quality Trend
          </h2>
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="label"
                  stroke="#9ca3af"
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  stroke="#9ca3af"
                  tick={{ fontSize: 11 }}
                  domain={[0, 10]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1f2937",
                    border: "1px solid #374151",
                    borderRadius: "0.5rem",
                    color: "#f3f4f6",
                  }}
                  formatter={(value: number) => [
                    value.toFixed(1),
                    "Avg Quality",
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="avgQuality"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={{ r: 4, fill: "#22c55e" }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="py-16 text-center text-gray-500">No data yet</p>
          )}
        </div>
      </div>

      {/* Pipeline Run Reports Table */}
      <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
        <h2 className="mb-4 text-lg font-semibold text-white">
          Pipeline Run Reports
        </h2>
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="h-12 animate-pulse rounded bg-gray-700/50"
              />
            ))}
          </div>
        ) : runs && runs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-gray-700 text-gray-400">
                <tr>
                  <th className="px-3 py-2 font-medium">ID</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Started</th>
                  <th className="px-3 py-2 font-medium text-center">
                    Accepted
                  </th>
                  <th className="px-3 py-2 font-medium text-center">Rejected</th>
                  <th className="px-3 py-2 font-medium text-center">
                    Pass Rate
                  </th>
                  <th className="px-3 py-2 font-medium text-center">
                    Avg Quality
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {runs.map((run: PipelineRun) => {
                  const total = run.posts_accepted + run.posts_rejected;
                  const passRate =
                    total > 0
                      ? Math.round((run.posts_accepted / total) * 100)
                      : 0;
                  return (
                    <tr key={run.run_id} className="hover:bg-gray-700/30">
                      <td className="px-3 py-2 text-gray-400">
                        #{run.run_id.slice(0, 8)}
                      </td>
                      <td className="px-3 py-2">
                        <PipelineStatus status={run.status} />
                      </td>
                      <td className="px-3 py-2 text-gray-300">
                        {new Date(run.started_at).toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-center text-gray-300">
                        {run.posts_accepted}
                      </td>
                      <td className="px-3 py-2 text-center text-gray-300">
                        {run.posts_rejected}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={
                            passRate >= 80
                              ? "text-green-400"
                              : passRate >= 50
                                ? "text-yellow-400"
                                : "text-red-400"
                          }
                        >
                          {passRate}%
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center text-gray-300">
                        {(run.avg_quality ?? 0).toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="py-8 text-center text-gray-500">No pipeline runs yet</p>
        )}
      </div>
    </div>
  );
}
