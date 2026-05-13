import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, Eye, MessageSquare, BarChart3 } from "lucide-react";
import { toggleUsed, type Post } from "../api/client";
import CopyButton from "./CopyButton";
import SourceBadge from "./SourceBadge";
import EvaluationReport from "./EvaluationReport";
import FeedbackPanel from "./FeedbackPanel";
import LinkedInPreview from "./LinkedInPreview";
import { StarRating } from "./FeedbackPanel";

type Tab = "text" | "preview" | "evaluation" | "feedback";

interface PostCardProps {
  post: Post;
}

export default function PostCard({ post }: PostCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [tab, setTab] = useState<Tab>("text");
  const queryClient = useQueryClient();

  const toggleMutation = useMutation({
    mutationFn: toggleUsed,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      queryClient.invalidateQueries({ queryKey: ["postStats"] });
    },
  });

  const scoreColor = (v: number) =>
    v >= 7 ? "text-green-400" : v >= 4 ? "text-yellow-400" : "text-red-400";

  const scoreBg = (v: number) =>
    v >= 7
      ? "bg-green-900/30 border-green-800"
      : v >= 4
        ? "bg-yellow-900/30 border-yellow-800"
        : "bg-red-900/30 border-red-800";

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 transition-colors hover:border-gray-600">
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        className="flex cursor-pointer items-center justify-between gap-4 px-5 py-4"
      >
        <div className="flex items-center gap-3 overflow-hidden">
          {expanded ? (
            <ChevronUp size={16} className="shrink-0 text-gray-500" />
          ) : (
            <ChevronDown size={16} className="shrink-0 text-gray-500" />
          )}
          <SourceBadge source={post.source} />
          <span className="truncate text-sm font-medium text-gray-200">
            {post.title || post.post_text.slice(0, 80)}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          {/* Human rating */}
          {(post.human_feedback_count ?? 0) > 0 && (
            <div className="flex items-center gap-1">
              <StarRating value={Math.round(post.human_rating_avg ?? 0)} readonly />
              <span className="text-[10px] text-gray-500">
                ({post.human_feedback_count})
              </span>
            </div>
          )}
          {/* Score display */}
          <div className={`rounded-lg border px-2.5 py-1 text-center ${scoreBg(post.quality_score)}`}>
            <p className="text-[10px] uppercase tracking-wider text-gray-400">Quality</p>
            <p className={`text-sm font-bold ${scoreColor(post.quality_score)}`}>
              {post.quality_score.toFixed(1)}
            </p>
          </div>
          <div className={`rounded-lg border px-2.5 py-1 text-center ${scoreBg(post.final_score)}`}>
            <p className="text-[10px] uppercase tracking-wider text-gray-400">Score</p>
            <p className={`text-sm font-bold ${scoreColor(post.final_score)}`}>
              {post.final_score.toFixed(1)}
            </p>
          </div>
          {/* Used toggle */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              toggleMutation.mutate(post.id);
            }}
            className={`h-6 w-11 rounded-full transition-colors ${
              post.is_used ? "bg-green-600" : "bg-gray-600"
            } relative`}
            title={post.is_used ? "Mark as unused" : "Mark as used"}
          >
            <span
              className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                post.is_used ? "left-5" : "left-0.5"
              }`}
            />
          </button>
        </div>
      </div>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-gray-700">
          {/* Tab bar */}
          <div className="flex border-b border-gray-700 px-5">
            {([
              { key: "text" as Tab, label: "Post Text", icon: null },
              { key: "preview" as Tab, label: "LinkedIn Preview", icon: Eye },
              { key: "evaluation" as Tab, label: "Evaluation", icon: BarChart3 },
              { key: "feedback" as Tab, label: "Feedback", icon: MessageSquare },
            ] as const).map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors ${
                  tab === key
                    ? "border-b-2 border-blue-500 text-blue-400"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {Icon && <Icon size={13} />}
                {label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="px-5 py-4">
            {tab === "text" && (
              <>
                <div className="flex items-start justify-between gap-4">
                  <pre className="flex-1 whitespace-pre-wrap text-sm leading-relaxed text-gray-300">
                    {post.post_text}
                  </pre>
                  <CopyButton text={post.post_text} />
                </div>
                <div className="mt-3 flex flex-wrap gap-4 text-xs text-gray-500">
                  <span>{post.char_count.toLocaleString()} characters</span>
                  {post.original_url && (
                    <a
                      href={post.original_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-blue-400 hover:underline"
                    >
                      Source link
                    </a>
                  )}
                  <span>{new Date(post.created_at).toLocaleString()}</span>
                  <span>Attempts: {post.generation_attempts}</span>
                </div>
              </>
            )}

            {tab === "preview" && <LinkedInPreview post={post} />}

            {tab === "evaluation" && <EvaluationReport postId={post.id} />}

            {tab === "feedback" && <FeedbackPanel postId={post.id} />}
          </div>
        </div>
      )}
    </div>
  );
}
