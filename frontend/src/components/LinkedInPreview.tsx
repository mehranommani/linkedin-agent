import { ThumbsUp, MessageCircle, Repeat2, Send } from "lucide-react";
import type { Post } from "../api/client";

export default function LinkedInPreview({ post }: { post: Post }) {
  const charCount = post.post_text?.length ?? 0;
  const charColor =
    charCount >= 1200 && charCount <= 3000
      ? "text-green-500"
      : charCount < 1200
        ? "text-yellow-500"
        : "text-red-500";

  // Parse hashtags from text
  const formattedText = post.post_text?.replace(
    /#(\w+)/g,
    '<span class="text-blue-400 font-medium">#$1</span>'
  );

  return (
    <div className="overflow-hidden rounded-xl border border-gray-600 bg-white shadow-lg">
      {/* LinkedIn Post Header */}
      <div className="flex items-center gap-3 px-4 pt-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-700 text-lg font-bold text-white">
          M
        </div>
        <div>
          <p className="text-sm font-semibold text-gray-900">Mehran</p>
          <p className="text-xs text-gray-500">AI/ML Engineer</p>
          <p className="text-xs text-gray-400">Just now</p>
        </div>
      </div>

      {/* Post Content */}
      <div className="px-4 py-3">
        <div
          className="whitespace-pre-wrap text-sm leading-relaxed text-gray-800"
          dangerouslySetInnerHTML={{ __html: formattedText || "" }}
        />
      </div>

      {/* Engagement Bar */}
      <div className="border-t border-gray-200 px-4 py-1">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>0 likes</span>
          <span>0 comments</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex border-t border-gray-200">
        {[
          { icon: ThumbsUp, label: "Like" },
          { icon: MessageCircle, label: "Comment" },
          { icon: Repeat2, label: "Repost" },
          { icon: Send, label: "Send" },
        ].map(({ icon: Icon, label }) => (
          <button
            key={label}
            className="flex flex-1 items-center justify-center gap-1.5 py-3 text-xs font-medium text-gray-600 transition-colors hover:bg-gray-50"
            disabled
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </div>

      {/* Character Count */}
      <div className="border-t border-gray-100 bg-gray-50 px-4 py-2 text-right">
        <span className={`text-xs font-medium ${charColor}`}>
          {charCount.toLocaleString()} chars
        </span>
        <span className="ml-2 text-xs text-gray-400">(1,200-3,000 optimal)</span>
      </div>
    </div>
  );
}
