import { useQuery } from "@tanstack/react-query";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";
import { getPostEvaluation, type DetailedEvaluation } from "../api/client";

const METRIC_LABELS: Record<string, string> = {
  answer_relevancy: "Relevancy",
  faithfulness: "Faithfulness",
  hallucination: "Hallucination",
  bias: "Bias",
  toxicity: "Toxicity",
  linkedin_quality: "LinkedIn Quality",
};

const THRESHOLDS: Record<string, number> = {
  answer_relevancy: 7.0,
  faithfulness: 7.0,
  hallucination: 8.0,
  bias: 7.0,
  toxicity: 9.0,
  linkedin_quality: 7.0,
};

function MetricBadge({ name, score }: { name: string; score: number }) {
  const threshold = THRESHOLDS[name] ?? 7.0;
  const passed = score >= threshold;
  return (
    <div
      className={`rounded-md border px-2 py-1 text-center text-xs ${
        passed
          ? "border-green-800 bg-green-900/30 text-green-400"
          : "border-red-800 bg-red-900/30 text-red-400"
      }`}
    >
      <span className="block text-[9px] uppercase tracking-wider text-gray-500">
        {METRIC_LABELS[name] ?? name}
      </span>
      <span className="font-bold">{score.toFixed(1)}</span>
    </div>
  );
}

interface EvaluationReportProps {
  postId: string;
  evaluation?: DetailedEvaluation | null;
}

export default function EvaluationReport({ postId, evaluation: evalProp }: EvaluationReportProps) {
  const { data: evalData } = useQuery({
    queryKey: ["evaluation", postId],
    queryFn: () => getPostEvaluation(postId),
    enabled: !evalProp && !!postId,
  });

  const evaluation = evalProp ?? evalData;

  if (!evaluation) {
    return (
      <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4 text-center text-sm text-gray-500">
        No detailed evaluation available
      </div>
    );
  }

  const metrics = [
    "answer_relevancy", "faithfulness", "hallucination",
    "bias", "toxicity", "linkedin_quality",
  ];

  const radarData = metrics.map((m) => ({
    metric: METRIC_LABELS[m] ?? m,
    score: (evaluation as unknown as Record<string, number>)[m] ?? 0,
    threshold: THRESHOLDS[m] ?? 7.0,
    fullMark: 10,
  }));

  return (
    <div className="space-y-4">
      {/* Overall Score Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`rounded-lg border px-3 py-2 text-center ${
              evaluation.passed
                ? "border-green-700 bg-green-900/30"
                : "border-red-700 bg-red-900/30"
            }`}
          >
            <span className="block text-[10px] uppercase tracking-wider text-gray-400">
              Overall
            </span>
            <span
              className={`text-xl font-bold ${
                evaluation.passed ? "text-green-400" : "text-red-400"
              }`}
            >
              {evaluation.overall_score.toFixed(1)}
            </span>
          </div>
          <span
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              evaluation.passed
                ? "bg-green-900/40 text-green-400"
                : "bg-red-900/40 text-red-400"
            }`}
          >
            {evaluation.passed ? "PASSED" : "FAILED"}
          </span>
        </div>
        <span className="text-xs text-gray-500">
          Judge: {evaluation.judge_model} ({evaluation.evaluation_version})
        </span>
      </div>

      {/* Metric Badges */}
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
        {metrics.map((m) => (
          <MetricBadge
            key={m}
            name={m}
            score={(evaluation as unknown as Record<string, number>)[m] ?? 0}
          />
        ))}
      </div>

      {/* Radar Chart */}
      <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-3">
        <ResponsiveContainer width="100%" height={250}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis
              dataKey="metric"
              tick={{ fill: "#9ca3af", fontSize: 11 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 10]}
              tick={{ fill: "#6b7280", fontSize: 10 }}
            />
            <Radar
              name="Threshold"
              dataKey="threshold"
              stroke="#f59e0b"
              fill="#f59e0b"
              fillOpacity={0.1}
              strokeDasharray="4 4"
            />
            <Radar
              name="Score"
              dataKey="score"
              stroke={evaluation.passed ? "#22c55e" : "#ef4444"}
              fill={evaluation.passed ? "#22c55e" : "#ef4444"}
              fillOpacity={0.25}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Issues & Strengths */}
      {evaluation.strengths?.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-medium uppercase tracking-wider text-green-400">
            Strengths
          </h4>
          <ul className="space-y-1">
            {evaluation.strengths.map((s, i) => (
              <li key={i} className="text-xs text-gray-400">+ {s}</li>
            ))}
          </ul>
        </div>
      )}
      {evaluation.issues?.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-medium uppercase tracking-wider text-red-400">
            Issues
          </h4>
          <ul className="space-y-1">
            {evaluation.issues.map((s, i) => (
              <li key={i} className="text-xs text-gray-400">- {s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export { MetricBadge, METRIC_LABELS, THRESHOLDS };
