import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { FileText, CheckCircle, Star, TrendingUp, MessageSquare, Heart } from "lucide-react";
import { getPostStats, getPosts, getFeedbackSummary, type PostStats, type Post } from "../api/client";
import PostCard from "../components/PostCard";

const SOURCE_COLORS: Record<string, string> = {
  github: "#6b7280",
  hackernews: "#f59e0b",
  reddit: "#ef4444",
  rss: "#3b82f6",
  arxiv: "#22c55e",
  devto: "#1f2937",
  producthunt: "#f59e0b",
  papers: "#a855f7",
};

function MetricCard({
  title,
  value,
  icon,
  color,
  subtitle,
}: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
}) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">{title}</p>
          <p className="mt-1 text-2xl font-bold text-white">{value}</p>
          {subtitle && (
            <p className="mt-0.5 text-xs text-gray-500">{subtitle}</p>
          )}
        </div>
        <div className={`rounded-lg p-3 ${color}`}>{icon}</div>
      </div>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-xl border border-gray-700 bg-gray-800 p-5">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <div className="h-4 w-24 rounded bg-gray-700" />
          <div className="h-7 w-16 rounded bg-gray-700" />
        </div>
        <div className="h-12 w-12 rounded-lg bg-gray-700" />
      </div>
    </div>
  );
}

function SkeletonChart() {
  return (
    <div className="animate-pulse rounded-xl border border-gray-700 bg-gray-800 p-6">
      <div className="mb-4 h-5 w-48 rounded bg-gray-700" />
      <div className="h-64 rounded bg-gray-700/50" />
    </div>
  );
}

export default function Dashboard() {
  const { data: stats, isLoading } = useQuery<PostStats>({
    queryKey: ["postStats"],
    queryFn: getPostStats,
  });

  const { data: recentData } = useQuery({
    queryKey: ["posts", { page: 1, per_page: 5, sort_by: "created_at" }],
    queryFn: () => getPosts({ page: 1, per_page: 5, sort_by: "created_at" }),
  });

  const { data: feedbackSummary } = useQuery({
    queryKey: ["feedbackSummary"],
    queryFn: getFeedbackSummary,
  });

  const chartData = stats
    ? stats.source_distribution.map((entry) => ({
        name: entry.source,
        count: entry.total,
      }))
    : [];

  const recentPosts = recentData?.posts ?? [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Dashboard</h1>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {isLoading ? (
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        ) : stats ? (
          <>
            <MetricCard
              title="Total Posts"
              value={stats.total_posts}
              icon={<FileText size={20} className="text-blue-400" />}
              color="bg-blue-900/40"
            />
            <MetricCard
              title="Used Posts"
              value={stats.used_posts}
              icon={<CheckCircle size={20} className="text-green-400" />}
              color="bg-green-900/40"
            />
            <MetricCard
              title="Avg Quality"
              value={(stats.avg_quality ?? 0).toFixed(1)}
              icon={<Star size={20} className="text-yellow-400" />}
              color="bg-yellow-900/40"
            />
            <MetricCard
              title="Avg Score"
              value={(stats.avg_score ?? 0).toFixed(1)}
              icon={<TrendingUp size={20} className="text-purple-400" />}
              color="bg-purple-900/40"
            />
            {feedbackSummary?.overall && (
              <>
                <MetricCard
                  title="Feedback"
                  value={feedbackSummary.overall.total_feedbacks}
                  subtitle={`${feedbackSummary.overall.unique_posts_rated} posts rated`}
                  icon={<MessageSquare size={20} className="text-cyan-400" />}
                  color="bg-cyan-900/40"
                />
                <MetricCard
                  title="Avg Rating"
                  value={(feedbackSummary.overall.avg_rating ?? 0).toFixed(1)}
                  subtitle="out of 5 stars"
                  icon={<Heart size={20} className="text-pink-400" />}
                  color="bg-pink-900/40"
                />
              </>
            )}
          </>
        ) : null}
      </div>

      {/* Source Distribution Chart + Top Rated */}
      <div className="grid gap-6 lg:grid-cols-2">
        {isLoading ? (
          <SkeletonChart />
        ) : (
          <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
            <h2 className="mb-4 text-lg font-semibold text-white">
              Source Distribution
            </h2>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1f2937",
                      border: "1px solid #374151",
                      borderRadius: "0.5rem",
                      color: "#f3f4f6",
                    }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell
                        key={index}
                        fill={SOURCE_COLORS[entry.name] ?? "#6b7280"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="py-12 text-center text-gray-500">No data yet</p>
            )}
          </div>
        )}

        {/* Top Rated Posts */}
        {feedbackSummary?.top_rated_posts && feedbackSummary.top_rated_posts.length > 0 && (
          <div className="rounded-xl border border-gray-700 bg-gray-800 p-6">
            <h2 className="mb-4 text-lg font-semibold text-white">
              Top Rated Posts
            </h2>
            <div className="space-y-3">
              {feedbackSummary.top_rated_posts.map((post) => (
                <div
                  key={post.id}
                  className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-900/50 px-4 py-3"
                >
                  <span className="truncate text-sm text-gray-200 mr-4">
                    {post.title}
                  </span>
                  <div className="flex items-center gap-2 shrink-0">
                    <div className="flex gap-0.5">
                      {[1, 2, 3, 4, 5].map((s) => (
                        <Star
                          key={s}
                          size={12}
                          className={
                            s <= Math.round(post.avg_rating)
                              ? "fill-yellow-400 text-yellow-400"
                              : "text-gray-600"
                          }
                        />
                      ))}
                    </div>
                    <span className="text-xs text-gray-500">
                      ({post.feedback_count})
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Recent Posts */}
      {recentPosts.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-white">Recent Posts</h2>
          {recentPosts.map((post: Post) => (
            <PostCard key={post.id} post={post} />
          ))}
        </div>
      )}
    </div>
  );
}
