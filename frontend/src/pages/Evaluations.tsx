import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { getPipelineRuns, type PipelineRun } from "../api/client";
import PipelineStatus from "../components/PipelineStatus";

export default function Evaluations() {
  const { data: runs, isLoading } = useQuery({
    queryKey: ["pipelineRuns"],
    queryFn: getPipelineRuns,
  });

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

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Evaluations</h1>

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
