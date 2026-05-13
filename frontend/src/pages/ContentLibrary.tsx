import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import {
  getPosts,
  getSources,
  type Post,
  type PostsParams,
} from "../api/client";
import PostCard from "../components/PostCard";
import Pagination from "../components/Pagination";

export default function ContentLibrary() {
  const [page, setPage] = useState(1);
  const [perPage] = useState(15);
  const [source, setSource] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("created_at");

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(searchInput);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput]);

  const params: PostsParams = {
    page,
    per_page: perPage,
    sort_by: sortBy,
    ...(source ? { source } : {}),
    ...(search ? { search } : {}),
  };

  const { data, isLoading, isPlaceholderData } = useQuery({
    queryKey: ["posts", params],
    queryFn: () => getPosts(params),
    placeholderData: (prev) => prev,
  });

  const { data: sources } = useQuery({
    queryKey: ["sources"],
    queryFn: getSources,
  });

  const uniqueSources = sources
    ? [...new Set(sources.map((s) => s.source_type))]
    : [];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-white">Content Library</h1>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[200px]">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
          />
          <input
            type="text"
            placeholder="Search posts..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 py-2 pl-9 pr-3 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
          />
        </div>

        <select
          value={source}
          onChange={(e) => {
            setSource(e.target.value);
            setPage(1);
          }}
          className="rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
        >
          <option value="">All Sources</option>
          {uniqueSources.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        <select
          value={sortBy}
          onChange={(e) => {
            setSortBy(e.target.value);
            setPage(1);
          }}
          className="rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
        >
          <option value="created_at">Newest First</option>
          <option value="quality_score">Highest Quality</option>
          <option value="final_score">Highest Score</option>
        </select>

        <span className="text-xs text-gray-500">
          {data?.total ?? 0} posts
        </span>
      </div>

      {/* Posts as Cards */}
      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="h-20 animate-pulse rounded-xl border border-gray-700 bg-gray-800"
            />
          ))}
        </div>
      ) : data?.posts && data.posts.length > 0 ? (
        <div className="space-y-3">
          {data.posts.map((post: Post) => (
            <PostCard key={post.id} post={post} />
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-gray-700 bg-gray-800 py-16 text-center text-gray-500">
          No posts found
        </div>
      )}

      {/* Pagination */}
      {data && data.pages > 1 && (
        <Pagination
          currentPage={data.page}
          totalPages={data.pages}
          onPageChange={setPage}
          disabled={isPlaceholderData}
        />
      )}
    </div>
  );
}
