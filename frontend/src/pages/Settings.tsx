import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Save, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import {
  getConfig,
  setConfig,
  getMcpTools,
  type ConfigEntry,
  type McpTool,
} from "../api/client";

function ConfigEntryRow({ entry }: { entry: ConfigEntry }) {
  const queryClient = useQueryClient();
  const isComplex =
    typeof entry.value === "object" && entry.value !== null;
  const [value, setValue] = useState(
    isComplex ? JSON.stringify(entry.value, null, 2) : String(entry.value ?? ""),
  );
  const [saved, setSaved] = useState(false);
  const [parseError, setParseError] = useState(false);

  const mutation = useMutation({
    mutationFn: (newValue: unknown) => setConfig(entry.key, newValue),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["config"] });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
  });

  const handleSave = () => {
    let parsed: unknown;
    if (isComplex) {
      try {
        parsed = JSON.parse(value);
        setParseError(false);
      } catch {
        setParseError(true);
        return;
      }
    } else if (value === "true" || value === "false") {
      parsed = value === "true";
    } else if (!isNaN(Number(value)) && value.trim() !== "") {
      parsed = Number(value);
    } else {
      parsed = value;
    }
    mutation.mutate(parsed);
  };

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4">
      <div className="mb-1 flex items-center justify-between">
        <div>
          <span className="text-sm font-medium text-gray-200">
            {entry.key}
          </span>
          {entry.description && (
            <p className="text-xs text-gray-500">{entry.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {saved && (
            <span className="flex items-center gap-1 text-xs text-green-400">
              <CheckCircle size={12} /> Saved
            </span>
          )}
          {parseError && (
            <span className="flex items-center gap-1 text-xs text-red-400">
              <AlertCircle size={12} /> Invalid JSON
            </span>
          )}
          <button
            onClick={handleSave}
            disabled={mutation.isPending}
            className="flex items-center gap-1 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? (
              <Loader2 size={12} className="animate-spin" />
            ) : (
              <Save size={12} />
            )}
            Save
          </button>
        </div>
      </div>
      {isComplex ? (
        <textarea
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setParseError(false);
          }}
          rows={Math.min(value.split("\n").length + 1, 12)}
          className={`mt-2 w-full rounded-lg border bg-gray-800 px-3 py-2 font-mono text-xs text-gray-200 focus:outline-none ${
            parseError
              ? "border-red-500 focus:border-red-500"
              : "border-gray-600 focus:border-blue-500"
          }`}
        />
      ) : (
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="mt-2 w-full rounded-lg border border-gray-600 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
        />
      )}
    </div>
  );
}

function McpToolRow({ tool }: { tool: McpTool }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-900/50 px-4 py-3">
      <div>
        <p className="text-sm font-medium text-gray-200">{tool.name}</p>
        <p className="text-xs text-gray-500">{tool.description}</p>
      </div>
      <span className="rounded-full bg-green-900/30 px-2.5 py-0.5 text-xs font-medium text-green-400">
        active
      </span>
    </div>
  );
}

export default function SettingsPage() {
  const { data: configs, isLoading: configLoading } = useQuery({
    queryKey: ["config"],
    queryFn: getConfig,
  });

  const { data: mcpTools, isLoading: toolsLoading } = useQuery({
    queryKey: ["mcpTools"],
    queryFn: getMcpTools,
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Settings</h1>

      {/* Config Editor */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Configuration</h2>
        {configLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="h-20 animate-pulse rounded-lg bg-gray-800"
              />
            ))}
          </div>
        ) : configs && configs.length > 0 ? (
          <div className="space-y-3">
            {configs.map((entry: ConfigEntry) => (
              <ConfigEntryRow key={entry.key} entry={entry} />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-gray-700 bg-gray-800 py-8 text-center text-gray-500">
            No configuration entries found
          </div>
        )}
      </div>

      {/* MCP Tools */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-white">MCP Tools Status</h2>
        {toolsLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className="h-14 animate-pulse rounded-lg bg-gray-800"
              />
            ))}
          </div>
        ) : mcpTools && mcpTools.length > 0 ? (
          <div className="space-y-2">
            {mcpTools.map((tool: McpTool) => (
              <McpToolRow key={tool.name} tool={tool} />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-gray-700 bg-gray-800 py-8 text-center text-gray-500">
            No MCP tools available
          </div>
        )}
      </div>
    </div>
  );
}
