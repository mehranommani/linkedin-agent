import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Star } from "lucide-react";
import {
  submitFeedback,
  getPostFeedback,
  type HumanFeedback,
  type HumanFeedbackCreate,
} from "../api/client";

function StarRating({
  value,
  onChange,
  readonly = false,
}: {
  value: number;
  onChange?: (v: number) => void;
  readonly?: boolean;
}) {
  const [hover, setHover] = useState(0);
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          disabled={readonly}
          onMouseEnter={() => !readonly && setHover(star)}
          onMouseLeave={() => !readonly && setHover(0)}
          onClick={() => onChange?.(star)}
          className={readonly ? "cursor-default" : "cursor-pointer"}
        >
          <Star
            size={readonly ? 14 : 20}
            className={
              star <= (hover || value)
                ? "fill-yellow-400 text-yellow-400"
                : "text-gray-600"
            }
          />
        </button>
      ))}
    </div>
  );
}

export default function FeedbackPanel({ postId }: { postId: string }) {
  const queryClient = useQueryClient();
  const [rating, setRating] = useState(0);
  const [text, setText] = useState("");
  const [isGood, setIsGood] = useState(false);

  const { data: feedbacks, isLoading: feedbacksLoading, refetch } = useQuery({
    queryKey: ["feedback", postId],
    queryFn: () => getPostFeedback(postId),
    staleTime: 0, // Always refetch to get latest feedback
  });

  const mutation = useMutation({
    mutationFn: (fb: HumanFeedbackCreate) => submitFeedback(postId, fb),
    onSuccess: async () => {
      // Immediately refetch feedback to show the new submission
      await refetch();
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      queryClient.invalidateQueries({ queryKey: ["feedbackSummary"] });
      setRating(0);
      setText("");
      setIsGood(false);
    },
  });

  const handleSubmit = () => {
    if (rating === 0) return;
    mutation.mutate({
      rating,
      text_feedback: text,
      is_good_example: isGood,
    });
  };

  return (
    <div className="space-y-4">
      {/* Submit Form */}
      <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4">
        <h4 className="mb-3 text-sm font-medium text-gray-300">Rate this post</h4>
        <div className="flex items-center gap-4">
          <StarRating value={rating} onChange={setRating} />
          {rating > 0 && (
            <span className="text-xs text-gray-500">{rating}/5</span>
          )}
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Optional: what did you like or dislike?"
          rows={2}
          className="mt-3 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-300 placeholder-gray-600 focus:border-blue-600 focus:outline-none"
        />
        <div className="mt-3 flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={isGood}
              onChange={(e) => setIsGood(e.target.checked)}
              className="rounded border-gray-600 bg-gray-800 text-blue-600"
            />
            Mark as good example
          </label>
          <button
            onClick={handleSubmit}
            disabled={rating === 0 || mutation.isPending}
            className="rounded-md bg-blue-600 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-50"
          >
            {mutation.isPending ? "Submitting..." : "Submit Feedback"}
          </button>
        </div>
      </div>

      {/* Existing Feedback */}
      <div className="space-y-2">
        <h4 className="text-xs font-medium uppercase tracking-wider text-gray-500">
          Previous Feedback {feedbacks && `(${feedbacks.length})`}
        </h4>
        {feedbacksLoading ? (
          <div className="rounded-md border border-gray-700 bg-gray-900/30 px-3 py-2 text-center text-xs text-gray-500">
            Loading feedback...
          </div>
        ) : feedbacks && feedbacks.length > 0 ? (
          feedbacks.map((fb: HumanFeedback) => (
            <div
              key={fb.id}
              className="rounded-md border border-gray-700 bg-gray-900/30 px-3 py-2"
            >
              <div className="flex items-center justify-between">
                <StarRating value={fb.rating} readonly />
                <span className="text-[10px] text-gray-600">
                  {new Date(fb.created_at).toLocaleDateString()}
                </span>
              </div>
              {fb.text_feedback && (
                <p className="mt-1 text-xs text-gray-400">{fb.text_feedback}</p>
              )}
              {fb.is_good_example && (
                <span className="mt-1 inline-block rounded-full bg-green-900/30 px-2 py-0.5 text-[10px] text-green-400">
                  Good example
                </span>
              )}
            </div>
          ))
        ) : (
          <div className="rounded-md border border-gray-700 bg-gray-900/30 px-3 py-2 text-center text-xs text-gray-500">
            No feedback yet. Be the first to rate this post!
          </div>
        )}
      </div>
    </div>
  );
}

export { StarRating };
