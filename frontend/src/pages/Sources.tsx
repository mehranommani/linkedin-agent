import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Plus,
  ChevronDown,
  ChevronRight,
  Pencil,
  Trash2,
  X,
  Save,
  AlertTriangle,
  ShieldAlert,
  Activity,
} from "lucide-react";
import {
  getSources,
  getSourceHealth,
  createSource,
  updateSource,
  deleteSource,
  type Source,
  type SourceCreate,
} from "../api/client";
import SourceBadge from "../components/SourceBadge";

const EMPTY_FORM: SourceCreate = {
  source_type: "rss",
  name: "",
  url: "",
  category: "",
  is_active: true,
  priority: 5,
};

export default function Sources() {
  const queryClient = useQueryClient();
  const [showModal, setShowModal] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<SourceCreate>(EMPTY_FORM);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(
    new Set(),
  );

  const { data: sources, isLoading } = useQuery({
    queryKey: ["sources"],
    queryFn: getSources,
  });

  const { data: healthData } = useQuery({
    queryKey: ["sourceHealth"],
    queryFn: getSourceHealth,
  });

  const createMut = useMutation({
    mutationFn: createSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      closeModal();
    },
  });

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<SourceCreate & { is_active: boolean; priority: number }> }) =>
      updateSource(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      closeModal();
    },
  });

  const deleteMut = useMutation({
    mutationFn: deleteSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const toggleActiveMut = useMutation({
    mutationFn: ({ id, is_active }: { id: string; is_active: boolean }) =>
      updateSource(id, { is_active }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const closeModal = () => {
    setShowModal(false);
    setEditingId(null);
    setForm(EMPTY_FORM);
  };

  const openAdd = () => {
    setForm(EMPTY_FORM);
    setEditingId(null);
    setShowModal(true);
  };

  const openEdit = (source: Source) => {
    setForm({
      source_type: source.source_type,
      name: source.name,
      url: source.url,
      category: source.category,
      is_active: source.is_active,
      priority: source.priority,
    });
    setEditingId(source.id);
    setShowModal(true);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (editingId) {
      updateMut.mutate({ id: editingId, data: form });
    } else {
      createMut.mutate(form);
    }
  };

  const handleDelete = (id: string) => {
    if (window.confirm("Delete this source?")) {
      deleteMut.mutate(id);
    }
  };

  const toggleGroup = (type: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  // Group by source_type
  const grouped = (sources ?? []).reduce<Record<string, Source[]>>(
    (acc, src) => {
      const key = src.source_type;
      if (!acc[key]) acc[key] = [];
      acc[key].push(src);
      return acc;
    },
    {},
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Sources</h1>
        <button
          onClick={openAdd}
          className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          <Plus size={16} />
          Add Source
        </button>
      </div>

      {/* Health Summary Banner */}
      {healthData?.summary && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border border-gray-700 bg-gray-800 p-3 text-center">
            <p className="text-xs text-gray-400">Total</p>
            <p className="text-xl font-bold text-white">{healthData.summary.total}</p>
          </div>
          <div className="rounded-lg border border-green-800 bg-green-900/20 p-3 text-center">
            <p className="text-xs text-green-400">Healthy</p>
            <p className="text-xl font-bold text-green-400">{healthData.summary.healthy}</p>
          </div>
          <div className="rounded-lg border border-red-800 bg-red-900/20 p-3 text-center">
            <p className="text-xs text-red-400">Unhealthy</p>
            <p className="text-xl font-bold text-red-400">{healthData.summary.unhealthy}</p>
          </div>
          <div className="rounded-lg border border-gray-600 bg-gray-900/30 p-3 text-center">
            <p className="text-xs text-gray-400">Disabled</p>
            <p className="text-xl font-bold text-gray-400">{healthData.summary.disabled}</p>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="h-16 animate-pulse rounded-xl bg-gray-800"
            />
          ))}
        </div>
      ) : Object.keys(grouped).length > 0 ? (
        <div className="space-y-4">
          {Object.entries(grouped).map(([type, srcs]) => {
            const isCollapsed = collapsedGroups.has(type);
            return (
              <div
                key={type}
                className="overflow-hidden rounded-xl border border-gray-700 bg-gray-800"
              >
                <button
                  onClick={() => toggleGroup(type)}
                  className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-gray-700/30"
                >
                  <div className="flex items-center gap-3">
                    {isCollapsed ? (
                      <ChevronRight size={16} className="text-gray-400" />
                    ) : (
                      <ChevronDown size={16} className="text-gray-400" />
                    )}
                    <SourceBadge source={type} />
                    <span className="text-sm text-gray-400">
                      ({srcs.length} source{srcs.length !== 1 ? "s" : ""})
                    </span>
                  </div>
                </button>
                {!isCollapsed && (
                  <table className="w-full text-left text-sm">
                    <thead className="border-y border-gray-700 bg-gray-800/50">
                      <tr>
                        <th className="px-4 py-2 font-medium text-gray-400">
                          Name
                        </th>
                        <th className="px-4 py-2 font-medium text-gray-400">
                          URL
                        </th>
                        <th className="px-4 py-2 font-medium text-gray-400">
                          Category
                        </th>
                        <th className="px-4 py-2 font-medium text-gray-400 text-center">
                          Active
                        </th>
                        <th className="px-4 py-2 font-medium text-gray-400 text-center">
                          Priority
                        </th>
                        <th className="px-4 py-2 font-medium text-gray-400 text-center">
                          Quality
                        </th>
                        <th className="px-4 py-2 font-medium text-gray-400 text-center">
                          Health
                        </th>
                        <th className="px-4 py-2 font-medium text-gray-400 text-center">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700">
                      {srcs.map((src) => (
                        <tr key={src.id} className="hover:bg-gray-700/20">
                          <td className="px-4 py-2 text-gray-200">
                            {src.name}
                          </td>
                          <td className="max-w-[200px] truncate px-4 py-2 text-gray-400">
                            <a
                              href={src.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="hover:text-blue-400 hover:underline"
                            >
                              {src.url}
                            </a>
                          </td>
                          <td className="px-4 py-2 text-gray-400">
                            {src.category || "-"}
                          </td>
                          <td className="px-4 py-2 text-center">
                            <button
                              onClick={() =>
                                toggleActiveMut.mutate({
                                  id: src.id,
                                  is_active: !src.is_active,
                                })
                              }
                              className={`h-5 w-10 rounded-full transition-colors ${
                                src.is_active ? "bg-green-600" : "bg-gray-600"
                              } relative`}
                            >
                              <span
                                className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
                                  src.is_active ? "left-5" : "left-0.5"
                                }`}
                              />
                            </button>
                          </td>
                          <td className="px-4 py-2 text-center text-gray-300">
                            {src.priority}
                          </td>
                          <td className="px-4 py-2 text-center text-gray-300">
                            {(src.avg_quality ?? 0).toFixed(1)}
                          </td>
                          <td className="px-4 py-2 text-center">
                            {(src.consecutive_failures ?? 0) > 0 ? (
                              <div className="flex items-center justify-center gap-1" title={src.last_error ?? ""}>
                                {(src.consecutive_failures ?? 0) >= 3 ? (
                                  <ShieldAlert size={14} className="text-red-400" />
                                ) : (
                                  <AlertTriangle size={14} className="text-yellow-400" />
                                )}
                                <span className={`text-xs ${(src.consecutive_failures ?? 0) >= 3 ? "text-red-400" : "text-yellow-400"}`}>
                                  {src.consecutive_failures}
                                </span>
                              </div>
                            ) : (
                              <Activity size={14} className="mx-auto text-green-400" />
                            )}
                          </td>
                          <td className="px-4 py-2 text-center">
                            <div className="flex items-center justify-center gap-1">
                              <button
                                onClick={() => openEdit(src)}
                                className="rounded p-1 text-gray-400 hover:bg-gray-600 hover:text-white"
                                title="Edit"
                              >
                                <Pencil size={14} />
                              </button>
                              <button
                                onClick={() => handleDelete(src.id)}
                                className="rounded p-1 text-gray-400 hover:bg-red-900/30 hover:text-red-400"
                                title="Delete"
                              >
                                <Trash2 size={14} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="rounded-xl border border-gray-700 bg-gray-800 py-16 text-center text-gray-500">
          No sources yet. Click "Add Source" to get started.
        </div>
      )}

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="w-full max-w-md rounded-xl border border-gray-700 bg-gray-800 p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">
                {editingId ? "Edit Source" : "Add Source"}
              </h3>
              <button
                onClick={closeModal}
                className="rounded p-1 text-gray-400 hover:bg-gray-700 hover:text-white"
              >
                <X size={18} />
              </button>
            </div>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="mb-1 block text-sm text-gray-400">
                  Source Type
                </label>
                <select
                  value={form.source_type}
                  onChange={(e) =>
                    setForm({ ...form, source_type: e.target.value })
                  }
                  className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                >
                  {[
                    "rss",
                    "github",
                    "hackernews",
                    "reddit",
                    "arxiv",
                    "devto",
                    "producthunt",
                    "papers",
                  ].map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm text-gray-400">Name</label>
                <input
                  type="text"
                  required
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                  placeholder="Source name"
                />
              </div>
              <div>
                <label className="mb-1 block text-sm text-gray-400">URL</label>
                <input
                  type="text"
                  required
                  value={form.url}
                  onChange={(e) => setForm({ ...form, url: e.target.value })}
                  className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                  placeholder="https://..."
                />
              </div>
              <div>
                <label className="mb-1 block text-sm text-gray-400">
                  Category
                </label>
                <input
                  type="text"
                  value={form.category ?? ""}
                  onChange={(e) =>
                    setForm({ ...form, category: e.target.value })
                  }
                  className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                  placeholder="e.g. ai, engineering"
                />
              </div>
              <div>
                <label className="mb-1 block text-sm text-gray-400">
                  Priority (1-10)
                </label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={form.priority ?? 5}
                  onChange={(e) =>
                    setForm({ ...form, priority: Number(e.target.value) })
                  }
                  className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is_active"
                  checked={form.is_active ?? true}
                  onChange={(e) =>
                    setForm({ ...form, is_active: e.target.checked })
                  }
                  className="h-4 w-4 rounded border-gray-600 bg-gray-700 accent-blue-500"
                />
                <label htmlFor="is_active" className="text-sm text-gray-300">
                  Active
                </label>
              </div>
              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={closeModal}
                  className="rounded-lg border border-gray-600 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={createMut.isPending || updateMut.isPending}
                  className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                >
                  <Save size={14} />
                  {editingId ? "Update" : "Create"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
