import { Loader2, CheckCircle, XCircle, Clock } from "lucide-react";

interface PipelineStatusProps {
  status: string;
}

export default function PipelineStatus({ status }: PipelineStatusProps) {
  const normalized = status.toLowerCase();

  if (normalized === "running" || normalized === "in_progress") {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-blue-400">
        <Loader2 size={14} className="animate-spin" />
        Running
      </span>
    );
  }

  if (normalized === "completed" || normalized === "success") {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-green-400">
        <CheckCircle size={14} />
        Completed
      </span>
    );
  }

  if (normalized === "failed" || normalized === "error") {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-red-400">
        <XCircle size={14} />
        Failed
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-gray-400">
      <Clock size={14} />
      {status}
    </span>
  );
}
