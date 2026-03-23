import { useState } from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import {
  LayoutDashboard,
  Library,
  Bot,
  BarChart3,
  Globe,
  Settings,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import Dashboard from "./pages/Dashboard";
import ContentLibrary from "./pages/ContentLibrary";
import AgentControl from "./pages/AgentControl";
import Evaluations from "./pages/Evaluations";
import Sources from "./pages/Sources";
import SettingsPage from "./pages/Settings";

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: <LayoutDashboard size={20} /> },
  { to: "/content", label: "Content Library", icon: <Library size={20} /> },
  { to: "/agent", label: "Agent Control", icon: <Bot size={20} /> },
  { to: "/evaluations", label: "Evaluations", icon: <BarChart3 size={20} /> },
  { to: "/sources", label: "Sources", icon: <Globe size={20} /> },
  { to: "/settings", label: "Settings", icon: <Settings size={20} /> },
];

export default function App() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <aside
        className={`${
          collapsed ? "w-16" : "w-60"
        } flex flex-col border-r border-gray-700 bg-gray-800 transition-all duration-200`}
      >
        {/* Brand */}
        <div className="flex h-14 items-center gap-2 border-b border-gray-700 px-4">
          <Bot size={24} className="shrink-0 text-blue-400" />
          {!collapsed && (
            <span className="text-lg font-semibold tracking-tight text-white">
              LinkedIn Agent
            </span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-2 py-4">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-blue-600/20 text-blue-400"
                    : "text-gray-400 hover:bg-gray-700 hover:text-gray-200"
                } ${collapsed ? "justify-center" : ""}`
              }
              title={collapsed ? item.label : undefined}
            >
              {item.icon}
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex h-10 items-center justify-center border-t border-gray-700 text-gray-400 hover:text-white"
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl p-6">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/content" element={<ContentLibrary />} />
            <Route path="/agent" element={<AgentControl />} />
            <Route path="/evaluations" element={<Evaluations />} />
            <Route path="/sources" element={<Sources />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
