import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Play, Loader2, Wifi, WifiOff, Trash2 } from "lucide-react";
import {
  startPipeline,
  getPipelineRuns,
  getSources,
  type PipelineRunConfig,
  type PipelineRun,
} from "../api/client";
import { usePipelineWebSocket } from "../api/websocket";
import PipelineStatus from "../components/PipelineStatus";

export default function AgentControl() {
  const queryClient = useQueryClient();
  const [maxPosts, setMaxPosts] = useState(5);
  const [maxAgeDays, setMaxAgeDays] = useState(7);
  const [maxRetries, setMaxRetries] = useState(3);
  const [selectedSourceTypes, setSelectedSourceTypes] = useState<string[]>([]);

  const { messages, isConnected, clearMessages } = usePipelineWebSocket();

  const { data: sources } = useQuery({
    queryKey: ["sources"],
    queryFn: getSources,
  });

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ["pipelineRuns"],
    queryFn: getPipelineRuns,
  });

  const runMutation = useMutation({
    mutationFn: startPipeline,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelineRuns"] });
    },
  });

  // Derive unique source types from loaded sources
  const sourceTypes = Array.from(
    new Set((sources ?? []).map((s) => s.source_type))
  ).sort();

  const handleToggleSourceType = (type: string) => {
    setSelectedSourceTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type],
    );
  };

  const handleRun = () => {
    clearMessages();
    const config: PipelineRunConfig = {
      max_posts: maxPosts,
      max_age_days: maxAgeDays,
      max_retries: maxRetries,
      sources: selectedSourceTypes.length > 0 ? selectedSourceTypes : undefined,
    };
    runMutation.mutate(config);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Agent Control</h1>
        <div className="flex items-center gap-2 text-sm">
          {isConnected ? (
            <span className="flex items-center gap-1 text-green-400">
              <Wifi size={14} /> Connected
            </span>
          ) : (
            <span className="flex items-center gap-1 text-red-400">
              <WifiOff size={14} /> Disconnected
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Config Form */}
        <div className="space-y-6 rounded-xl border border-gray-700 bg-gray-800 p-6">
          <h2 className="text-lg font-semibold text-white">Pipeline Config</h2>

          {/* Max Posts */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">
              Max Posts: <span className="font-medium text-white">{maxPosts}</span>
            </label>
            <input
              type="range"
              min={1}
              max={20}
              value={maxPosts}
              onChange={(e) => setMaxPosts(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>20</span>
            </div>
          </div>

          {/* Max Age Days */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">
              Max Age (days):{" "}
              <span className="font-medium text-white">{maxAgeDays}</span>
            </label>
            <input
              type="range"
              min={1}
              max={30}
              value={maxAgeDays}
              onChange={(e) => setMaxAgeDays(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>30</span>
            </div>
          </div>

          {/* Max Retries */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">
              Max Retries:{" "}
              <span className="font-medium text-white">{maxRetries}</span>
            </label>
            <input
              type="range"
              min={1}
              max={5}
              value={maxRetries}
              onChange={(e) => setMaxRetries(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>5</span>
            </div>
          </div>

          {/* Source Type Filter */}
          <div>
            <div className="mb-2 flex items-center justify-between">
              <label className="text-sm text-gray-400">
                Sources{" "}
                <span className="text-gray-600 text-xs">
                  {selectedSourceTypes.length === 0
                    ? "(all active)"
                    : `(${selectedSourceTypes.join(", ")})`}
                </span>
              </label>
              {selectedSourceTypes.length > 0 && (
                <button
                  onClick={() => setSelectedSourceTypes([])}
                  className="text-xs text-gray-500 hover:text-gray-300"
                >
                  clear
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {sourceTypes.map((type) => (
                <button
                  key={type}
                  type="button"
                  onClick={() => handleToggleSourceType(type)}
                  className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${
                    selectedSourceTypes.includes(type)
                      ? "border-blue-500 bg-blue-900/30 text-blue-300"
                      : "border-gray-600 bg-gray-700/50 text-gray-400 hover:border-gray-500 hover:text-gray-200"
                  }`}
                >
                  {type}
                </button>
              ))}
              {sourceTypes.length === 0 && (
                <p className="text-sm text-gray-500">No sources configured</p>
              )}
            </div>
          </div>

          {/* Run Button */}
          <button
            onClick={handleRun}
            disabled={runMutation.isPending}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
          >
            {runMutation.isPending ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Play size={16} />
            )}
            {runMutation.isPending ? "Running..." : "Run Pipeline"}
          </button>

          {runMutation.isError && (
            <p className="text-sm text-red-400">
              Error: {(runMutation.error as Error).message}
            </p>
          )}
        </div>

        {/* Live Console */}
        <div className="flex flex-col rounded-xl border border-gray-700 bg-gray-800">
          <div className="flex items-center justify-between border-b border-gray-700 px-4 py-3">
            <h2 className="text-lg font-semibold text-white">Live Console</h2>
            <button
              onClick={clearMessages}
              className="rounded p-1 text-gray-400 hover:bg-gray-700 hover:text-white"
              title="Clear"
            >
              <Trash2 size={14} />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-relaxed">
            {messages.length === 0 ? (
              <p className="text-gray-500">
                Waiting for pipeline messages...
              </p>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className="mb-1">
                  <span className="text-gray-500">
                    [{new Date(msg.timestamp).toLocaleTimeString()}]
                  </span>{" "}
                  <span
                    className={
                      msg.type === "error"
                        ? "text-red-400"
                        : msg.type === "complete"
                          ? "text-green-400"
                          : "text-gray-300"
                    }
                  >
                    {msg.message}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Pipeline Run History */}
      <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
        <h2 className="mb-4 text-lg font-semibold text-white">
          Pipeline Run History
        </h2>
        {runsLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
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
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Started</th>
                  <th className="px-3 py-2 font-medium">Finished</th>
                  <th className="px-3 py-2 font-medium text-center">
                    Accepted
                  </th>
                  <th className="px-3 py-2 font-medium text-center">Rejected</th>
                  <th className="px-3 py-2 font-medium text-center">
                    Avg Quality
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {runs.map((run: PipelineRun) => (
                  <tr key={run.run_id} className="hover:bg-gray-700/30">
                    <td className="px-3 py-2">
                      <PipelineStatus status={run.status} />
                    </td>
                    <td className="px-3 py-2 text-gray-300">
                      {new Date(run.started_at).toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-gray-400">
                      {run.completed_at
                        ? new Date(run.completed_at).toLocaleString()
                        : "-"}
                    </td>
                    <td className="px-3 py-2 text-center text-gray-300">
                      {run.posts_accepted}
                    </td>
                    <td className="px-3 py-2 text-center text-gray-300">
                      {run.posts_rejected}
                    </td>
                    <td className="px-3 py-2 text-center text-gray-300">
                      {(run.avg_quality ?? 0).toFixed(1)}
                    </td>
                  </tr>
                ))}
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
