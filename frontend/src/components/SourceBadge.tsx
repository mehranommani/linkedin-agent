interface SourceBadgeProps {
  source: string;
}

const SOURCE_STYLES: Record<string, string> = {
  github: "bg-gray-700 text-gray-300",
  hackernews: "bg-orange-900/40 text-orange-400",
  reddit: "bg-red-900/40 text-red-400",
  rss: "bg-blue-900/40 text-blue-400",
  arxiv: "bg-green-900/40 text-green-400",
  devto: "bg-gray-900 text-gray-300 border border-gray-600",
  producthunt: "bg-amber-900/40 text-amber-400",
  papers: "bg-purple-900/40 text-purple-400",
};

export default function SourceBadge({ source }: SourceBadgeProps) {
  const style =
    SOURCE_STYLES[source.toLowerCase()] ??
    "bg-gray-700 text-gray-300";

  return (
    <span
      className={`inline-flex shrink-0 items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${style}`}
    >
      {source}
    </span>
  );
}
