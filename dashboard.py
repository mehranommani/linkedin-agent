"""
LinkedIn Content Studio
=======================
Professional dashboard for AI-powered LinkedIn content generation.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import csv
import subprocess
import threading
import queue
import time
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LinkedIn Content Studio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "linkedin_content_bank.csv"
AGENT_FILE = BASE_DIR / "agent.py"
LOG_FILE = BASE_DIR / ".agent_output.log"

# ============================================================
# SOURCE CONFIGURATION
# ============================================================

SOURCE_CONFIG = {
    "github": {"name": "GitHub", "icon": "🐙", "color": "#24292e", "gradient": "linear-gradient(135deg, #24292e 0%, #404448 100%)"},
    "hackernews": {"name": "Hacker News", "icon": "🟠", "color": "#ff6600", "gradient": "linear-gradient(135deg, #ff6600 0%, #ff8533 100%)"},
    "reddit": {"name": "Reddit", "icon": "🔴", "color": "#ff4500", "gradient": "linear-gradient(135deg, #ff4500 0%, #ff6b3d 100%)"},
    "producthunt": {"name": "Product Hunt", "icon": "🚀", "color": "#da552f", "gradient": "linear-gradient(135deg, #da552f 0%, #ea7b56 100%)"},
    "papers": {"name": "Papers", "icon": "📄", "color": "#21cbce", "gradient": "linear-gradient(135deg, #21cbce 0%, #4dd9db 100%)"},
    "arxiv": {"name": "Arxiv", "icon": "🔬", "color": "#b31b1b", "gradient": "linear-gradient(135deg, #b31b1b 0%, #d44444 100%)"},
    "devto": {"name": "Dev.to", "icon": "📝", "color": "#0a0a0a", "gradient": "linear-gradient(135deg, #0a0a0a 0%, #333333 100%)"},
    "rss": {"name": "RSS Feeds", "icon": "📡", "color": "#ee802f", "gradient": "linear-gradient(135deg, #ee802f 0%, #f5a053 100%)"}
}

# ============================================================
# PROFESSIONAL CSS DESIGN
# ============================================================

st.markdown("""
<style>
    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary: #0a66c2;
        --primary-dark: #004182;
        --primary-light: #70b5f9;
        --success: #057642;
        --warning: #915907;
        --danger: #cc1016;
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-400: #94a3b8;
        --gray-500: #64748b;
        --gray-600: #475569;
        --gray-700: #334155;
        --gray-800: #1e293b;
        --gray-900: #0f172a;
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }

    /* ===== GLOBAL STYLES ===== */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ===== HEADER ===== */
    .studio-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 2.5rem 3rem;
        border-radius: var(--radius-xl);
        margin-bottom: 2rem;
        color: white;
        box-shadow: var(--shadow-xl);
    }
    .studio-header h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    .studio-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }

    /* ===== METRIC CARDS ===== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--gray-100);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--gray-900);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.875rem;
        color: var(--gray-500);
        margin-top: 0.25rem;
    }
    .metric-delta {
        font-size: 0.75rem;
        padding: 0.125rem 0.5rem;
        border-radius: 999px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .delta-positive {
        background: #dcfce7;
        color: var(--success);
    }
    .delta-neutral {
        background: var(--gray-100);
        color: var(--gray-600);
    }

    /* ===== SOURCE STATS ===== */
    .source-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin: 1.5rem 0;
    }
    .source-stat-card {
        background: white;
        border-radius: var(--radius-md);
        padding: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--gray-100);
        transition: all 0.2s ease;
    }
    .source-stat-card:hover {
        border-color: var(--primary-light);
    }
    .source-icon {
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: var(--radius-sm);
    }
    .source-info .count {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--gray-900);
    }
    .source-info .name {
        font-size: 0.75rem;
        color: var(--gray-500);
    }

    /* ===== POST CARDS ===== */
    .post-card {
        background: white;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--gray-100);
        transition: all 0.2s ease;
    }
    .post-card:hover {
        box-shadow: var(--shadow-lg);
    }
    .post-card.used {
        opacity: 0.65;
        border-left: 4px solid var(--gray-400);
    }
    .post-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }
    .post-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--gray-900);
        margin: 0;
        line-height: 1.4;
    }
    .post-meta {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: var(--gray-500);
    }

    /* ===== BADGES ===== */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.625rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    .badge-source {
        color: white;
    }
    .badge-score-high {
        background: #dcfce7;
        color: var(--success);
    }
    .badge-score-medium {
        background: #fef3c7;
        color: var(--warning);
    }
    .badge-score-low {
        background: #fee2e2;
        color: var(--danger);
    }
    .badge-trending {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .badge-used {
        background: var(--gray-200);
        color: var(--gray-600);
    }

    /* ===== LINKEDIN PREVIEW ===== */
    .linkedin-preview {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: #e8e8e8;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #2a2a4a;
    }

    /* ===== CHAR COUNT ===== */
    .char-optimal { color: var(--success); font-weight: 600; }
    .char-warning { color: var(--warning); font-weight: 600; }
    .char-danger { color: var(--danger); font-weight: 600; }

    /* ===== AGENT CONSOLE ===== */
    .agent-console {
        background: #0d1117;
        color: #58a6ff;
        padding: 1.25rem;
        border-radius: var(--radius-md);
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 12px;
        line-height: 1.6;
        max-height: 450px;
        overflow-y: auto;
        white-space: pre-wrap;
        border: 1px solid #30363d;
    }
    .agent-console .success { color: #3fb950; }
    .agent-console .warning { color: #d29922; }
    .agent-console .error { color: #f85149; }

    /* ===== CONTROL PANEL ===== */
    .control-panel {
        background: white;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--gray-100);
    }
    .control-section {
        margin-bottom: 1.5rem;
    }
    .control-section:last-child {
        margin-bottom: 0;
    }
    .control-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--gray-700);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ===== SOURCE SELECTOR ===== */
    .source-selector {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
    }
    .source-option {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.625rem 0.75rem;
        background: var(--gray-50);
        border-radius: var(--radius-sm);
        border: 1px solid var(--gray-200);
        font-size: 0.8rem;
        transition: all 0.15s ease;
    }
    .source-option:hover {
        border-color: var(--primary);
        background: white;
    }
    .source-option.selected {
        border-color: var(--primary);
        background: #eff6ff;
    }

    /* ===== RUN BUTTON ===== */
    .run-button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 0.875rem 1.5rem;
        border-radius: var(--radius-md);
        font-weight: 600;
        font-size: 1rem;
        border: none;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    .run-button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }
    .run-button:disabled {
        background: var(--gray-300);
        cursor: not-allowed;
        transform: none;
    }

    /* ===== STATUS INDICATOR ===== */
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-ready {
        background: #dcfce7;
        color: var(--success);
    }
    .status-running {
        background: #dbeafe;
        color: var(--primary);
    }
    .status-error {
        background: #fee2e2;
        color: var(--danger);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    .status-dot.ready { background: var(--success); }
    .status-dot.running { background: var(--primary); }
    .status-dot.error { background: var(--danger); }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--gray-100);
        padding: 0.5rem;
        border-radius: var(--radius-lg);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-md);
        padding: 0.625rem 1.25rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: var(--shadow-sm);
    }

    /* ===== SIDEBAR ===== */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: var(--gray-50);
    }
    .sidebar-section {
        background: white;
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--gray-500);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    /* ===== EMPTY STATE ===== */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: var(--gray-50);
        border-radius: var(--radius-xl);
        border: 2px dashed var(--gray-200);
    }
    .empty-state h2 {
        color: var(--gray-700);
        margin-bottom: 0.5rem;
    }
    .empty-state p {
        color: var(--gray-500);
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--gray-100);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--gray-300);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gray-400);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# AGENT RUNNER (DevOps Implementation)
# ============================================================

class AgentRunner:
    """Thread-safe agent runner with output streaming."""

    def __init__(self):
        self.output_queue = queue.Queue()
        self.is_running = False
        self.return_code = None
        self.thread = None

    def run_agent(self, max_posts: int, max_age_days: int, sources: list = None):
        """Start agent in background thread."""
        if self.is_running:
            return False

        self.is_running = True
        self.return_code = None

        # Clear queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        self.thread = threading.Thread(
            target=self._execute_agent,
            args=(max_posts, max_age_days, sources),
            daemon=True
        )
        self.thread.start()
        return True

    def _execute_agent(self, max_posts: int, max_age_days: int, sources: list = None):
        """Execute agent and stream output to queue."""
        try:
            cmd = ["python3", str(AGENT_FILE), str(max_posts), str(max_age_days)]
            if sources:
                cmd.append(",".join(sources))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(BASE_DIR)
            )

            for line in iter(process.stdout.readline, ''):
                if line:
                    self.output_queue.put(line)

            process.wait()
            self.return_code = process.returncode

            if self.return_code == 0:
                self.output_queue.put("\n✅ Agent completed successfully!\n")
            else:
                self.output_queue.put(f"\n❌ Agent exited with code {self.return_code}\n")

        except Exception as e:
            self.output_queue.put(f"\n❌ Error: {str(e)}\n")
            self.return_code = -1
        finally:
            self.is_running = False

    def get_output(self) -> list:
        """Get all available output lines."""
        lines = []
        while not self.output_queue.empty():
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines


# Initialize agent runner in session state
if 'agent_runner' not in st.session_state:
    st.session_state.agent_runner = AgentRunner()
if 'agent_output_lines' not in st.session_state:
    st.session_state.agent_output_lines = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_source_badge_html(source: str) -> str:
    """Generate source badge HTML."""
    config = SOURCE_CONFIG.get(source, {"name": source.title(), "icon": "📌", "gradient": "linear-gradient(135deg, #666 0%, #888 100%)"})
    return f'<span class="badge badge-source" style="background: {config["gradient"]}">{config["icon"]} {config["name"]}</span>'


def get_score_badge_html(score: float) -> str:
    """Generate score badge HTML."""
    if score >= 8:
        return f'<span class="badge badge-score-high">★ {score:.1f}</span>'
    elif score >= 5:
        return f'<span class="badge badge-score-medium">★ {score:.1f}</span>'
    return f'<span class="badge badge-score-low">★ {score:.1f}</span>'


def load_data():
    """Load and prepare data."""
    if not CSV_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(CSV_FILE, on_bad_lines='skip', quoting=csv.QUOTE_ALL)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

        if 'linkedin_post' in df.columns:
            df['char_count'] = df['linkedin_post'].str.len()

        if 'used' not in df.columns:
            df['used'] = False
        df['used'] = df['used'].fillna(False).astype(bool)

        return df.sort_values(by='date', ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def mark_post_used(original_link: str, used: bool = True):
    """Mark post as used/unused."""
    try:
        df = pd.read_csv(CSV_FILE, on_bad_lines='skip', quoting=csv.QUOTE_ALL)
        if 'used' not in df.columns:
            df['used'] = False
        df.loc[df['original_link'] == original_link, 'used'] = used
        df.to_csv(CSV_FILE, index=False, quoting=csv.QUOTE_ALL)
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False


def get_char_status(count: int) -> tuple:
    """Get character count status."""
    if 1400 <= count <= 2100:
        return "Optimal", "char-optimal"
    elif 1000 <= count < 1400 or 2100 < count <= 2500:
        return "Warning", "char-warning"
    return "Review", "char-danger"


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    """Render studio header."""
    st.markdown("""
    <div class="studio-header">
        <h1>💼 LinkedIn Content Studio</h1>
        <p>AI-powered content generation and management platform</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(df: pd.DataFrame):
    """Render metrics dashboard."""
    total = len(df)
    unused = len(df[~df['used']]) if 'used' in df.columns else total
    avg_score = df['final_score'].mean() if 'final_score' in df.columns else 0
    optimal = len(df[(df['char_count'] >= 1400) & (df['char_count'] <= 2100)]) if 'char_count' in df.columns else 0
    trending = len(df[df['trending_boost'] > 0]) if 'trending_boost' in df.columns else 0

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Posts</div>
            <span class="metric-delta delta-neutral">All time</span>
        </div>
        <div class="metric-card">
            <div class="metric-value">{unused}</div>
            <div class="metric-label">Ready to Use</div>
            <span class="metric-delta delta-positive">{unused/total*100:.0f}% available</span>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Avg Score</div>
            <span class="metric-delta {'delta-positive' if avg_score >= 7 else 'delta-neutral'}">{'Good' if avg_score >= 7 else 'Fair'}</span>
        </div>
        <div class="metric-card">
            <div class="metric-value">{optimal}</div>
            <div class="metric-label">Optimal Length</div>
            <span class="metric-delta delta-positive">{optimal/total*100:.0f}%</span>
        </div>
        <div class="metric-card">
            <div class="metric-value">{trending}</div>
            <div class="metric-label">Trending</div>
            <span class="metric-delta delta-positive">🔥 Hot topics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_source_stats(df: pd.DataFrame):
    """Render source distribution."""
    if 'source' not in df.columns or len(df) == 0:
        return

    counts = df['source'].value_counts()

    # Use st.columns for proper Streamlit rendering
    cols = st.columns(min(len(counts), 4))

    for idx, (source, count) in enumerate(counts.items()):
        config = SOURCE_CONFIG.get(source, {"name": source.title(), "icon": "📌", "color": "#666"})
        with cols[idx % 4]:
            st.markdown(f"""
                <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.25rem 0;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid {config['color']};">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{config['icon']} {count}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">{config['name']}</div>
                </div>
            """, unsafe_allow_html=True)


def render_post_card(row: pd.Series, index: int):
    """Render a post card."""
    is_used = row.get('used', False)
    source = row.get('source', 'unknown')

    with st.container():
        # Build badges
        badges = get_source_badge_html(source)
        if is_used:
            badges += ' <span class="badge badge-used">✓ Used</span>'
        if 'final_score' in row and pd.notna(row.get('final_score')):
            badges += ' ' + get_score_badge_html(row['final_score'])
        if row.get('trending_boost', 0) > 0:
            badges += ' <span class="badge badge-trending">🔥 Trending</span>'

        # Header
        col1, col2 = st.columns([5, 1])
        with col1:
            title = row['title'][:85] + '...' if len(row['title']) > 85 else row['title']
            date_str = row['date'].strftime('%b %d, %Y') if pd.notna(row['date']) else 'Unknown'

            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">{badges}</div>
            <h4 style="margin: 0; font-size: 1rem; color: {'#64748b' if is_used else '#1e293b'};">
                {'~~' + title + '~~' if is_used else title}
            </h4>
            <div class="post-meta">
                <span>📅 {date_str}</span>
                <span>•</span>
                <a href="{row['original_link']}" target="_blank" style="color: #0a66c2; text-decoration: none;">View Source ↗</a>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            btn_label = "↩️ Restore" if is_used else "✓ Mark Used"
            if st.button(btn_label, key=f"toggle_{index}", use_container_width=True):
                mark_post_used(row['original_link'], not is_used)
                st.rerun()

        # Content tabs
        tab1, tab2, tab3 = st.tabs(["📋 Copy", "👁️ Preview", "📊 Stats"])

        with tab1:
            post_text = str(row.get('linkedin_post', ''))

            # Copy button with JavaScript
            copy_btn_id = f"copy_btn_{index}"
            textarea_id = f"textarea_{index}"

            st.markdown(f"""
            <style>
                .copy-btn {{
                    background: linear-gradient(135deg, #0a66c2 0%, #004182 100%);
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    font-size: 0.875rem;
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    transition: all 0.2s ease;
                    margin-bottom: 0.75rem;
                }}
                .copy-btn:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(10, 102, 194, 0.3);
                }}
                .copy-btn.copied {{
                    background: linear-gradient(135deg, #057642 0%, #03582f 100%);
                }}
            </style>
            <button class="copy-btn" id="{copy_btn_id}" onclick="
                const text = document.getElementById('{textarea_id}').value;
                navigator.clipboard.writeText(text).then(() => {{
                    const btn = document.getElementById('{copy_btn_id}');
                    btn.innerHTML = '✅ Copied!';
                    btn.classList.add('copied');
                    setTimeout(() => {{
                        btn.innerHTML = '📋 Copy to Clipboard';
                        btn.classList.remove('copied');
                    }}, 2000);
                }});
            ">📋 Copy to Clipboard</button>
            """, unsafe_allow_html=True)

            st.text_area("", value=post_text, height=300, key=f"copy_{index}", label_visibility="collapsed")

            # Hidden textarea for JS copy (since Streamlit text_area has complex DOM)
            st.markdown(f'<textarea id="{textarea_id}" style="position: absolute; left: -9999px;">{post_text}</textarea>', unsafe_allow_html=True)

            char_count = len(post_text)
            status, status_class = get_char_status(char_count)
            st.caption(f"Characters: **{char_count:,}** • Status: <span class='{status_class}'>{status}</span>", unsafe_allow_html=True)

        with tab2:
            st.markdown(f"""
            <div class="linkedin-preview">{row.get('linkedin_post', '')}</div>
            """, unsafe_allow_html=True)

        with tab3:
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("Characters", f"{len(str(row.get('linkedin_post', ''))):,}")
            with stat_cols[1]:
                st.metric("Relevance", f"{row.get('relevance_score', 0):.1f}/10")
            with stat_cols[2]:
                st.metric("Quality", f"{row.get('quality_score', 0):.1f}/10")

        st.divider()


def render_agent_control():
    """Render agent control panel."""
    runner = st.session_state.agent_runner

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### ⚙️ Configuration")

        max_posts = st.slider("Posts to generate", 1, 30, 10, help="Number of LinkedIn posts to create")
        max_age_days = st.slider("Content age (days)", 1, 30, 7, help="Fetch content from last N days")

        st.markdown("#### 📡 Sources")

        # Source selection as checkboxes
        selected_sources = {}
        source_cols = st.columns(2)
        for idx, (key, config) in enumerate(SOURCE_CONFIG.items()):
            with source_cols[idx % 2]:
                selected_sources[key] = st.checkbox(
                    f"{config['icon']} {config['name']}",
                    value=True,
                    key=f"src_{key}"
                )

    with col2:
        st.markdown("#### 🚀 Execute Agent")

        enabled_count = sum(selected_sources.values())

        # Status indicator
        if runner.is_running:
            st.markdown("""
            <div class="status-indicator status-running">
                <span class="status-dot running"></span>
                Agent is running...
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-indicator status-ready">
                <span class="status-dot ready"></span>
                Ready • {enabled_count} sources • {max_posts} posts • {max_age_days} days
            </div>
            """, unsafe_allow_html=True)

        # Run button
        if st.button(
            "🚀 Run Agent" if not runner.is_running else "⏳ Running...",
            type="primary",
            disabled=runner.is_running or enabled_count == 0,
            use_container_width=True
        ):
            # Get list of enabled sources
            enabled_sources = [key for key, enabled in selected_sources.items() if enabled]
            st.session_state.agent_output_lines = []
            runner.run_agent(max_posts, max_age_days, enabled_sources)
            st.rerun()

        # Console output
        st.markdown("#### 📟 Console Output")

        # Get new output
        if runner.is_running or runner.return_code is not None:
            new_lines = runner.get_output()
            st.session_state.agent_output_lines.extend(new_lines)

        # Display output
        output_text = ''.join(st.session_state.agent_output_lines[-100:])  # Last 100 lines
        if output_text:
            st.markdown(f'<div class="agent-console">{output_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="agent-console">Waiting for output...</div>', unsafe_allow_html=True)

        # Auto-refresh while running
        if runner.is_running:
            time.sleep(0.5)
            st.rerun()

        # Show completion message
        if runner.return_code is not None and not runner.is_running:
            if runner.return_code == 0:
                st.success("✅ Agent completed! Refresh the Posts tab to see new content.")
            else:
                st.error(f"Agent failed with code {runner.return_code}")

        st.markdown("---")
        st.markdown("**Manual command:**")
        enabled_sources_list = [key for key, enabled in selected_sources.items() if enabled]
        sources_arg = ",".join(enabled_sources_list) if enabled_sources_list else "all"
        st.code(f"python3 agent.py {max_posts} {max_age_days} {sources_arg}", language="bash")


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters."""
    st.sidebar.markdown("## 🎛️ Filters")

    # Search
    search = st.sidebar.text_input("🔍 Search", placeholder="Keywords...")

    # Source filter
    if 'source' in df.columns and len(df) > 0:
        sources = ["All"] + df['source'].unique().tolist()
        selected_source = st.sidebar.selectbox("📡 Source", sources)
    else:
        selected_source = "All"

    # Date filter
    if len(df) > 0:
        dates = ["All"] + sorted([str(d) for d in df['date'].dt.date.unique()], reverse=True)
        selected_date = st.sidebar.selectbox("📅 Date", dates)
    else:
        selected_date = "All"

    # Status filter
    status_filter = st.sidebar.radio("✓ Status", ["All", "Unused", "Used"])

    # Length filter
    length_filter = st.sidebar.radio("📏 Length", ["All", "Optimal", "Short", "Long"])

    # Apply filters
    filtered = df.copy()

    if search:
        mask = (
            filtered['title'].str.contains(search, case=False, na=False) |
            filtered['linkedin_post'].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    if selected_source != "All":
        filtered = filtered[filtered['source'] == selected_source]

    if selected_date != "All":
        filtered = filtered[filtered['date'].dt.date == pd.to_datetime(selected_date).date()]

    if 'used' in filtered.columns:
        if status_filter == "Unused":
            filtered = filtered[~filtered['used']]
        elif status_filter == "Used":
            filtered = filtered[filtered['used']]

    if 'char_count' in filtered.columns:
        if length_filter == "Optimal":
            filtered = filtered[(filtered['char_count'] >= 1400) & (filtered['char_count'] <= 2100)]
        elif length_filter == "Short":
            filtered = filtered[filtered['char_count'] < 1400]
        elif length_filter == "Long":
            filtered = filtered[filtered['char_count'] > 2100]

    # Stats
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(filtered)} of {len(df)}**")

    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    return filtered


# ============================================================
# MAIN APP
# ============================================================

def main():
    render_header()

    tab_posts, tab_agent, tab_sources = st.tabs(["📝 Content Library", "🤖 Generate New", "📡 Sources"])

    with tab_posts:
        df = load_data()

        if df.empty:
            st.markdown("""
            <div class="empty-state">
                <h2>📭 No Content Yet</h2>
                <p>Go to "Generate New" to create your first LinkedIn posts</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            filtered_df = render_sidebar(df)
            render_metrics(df)

            st.markdown("### 📊 Distribution by Source")
            render_source_stats(df)

            st.markdown(f"### 📝 Posts ({len(filtered_df)})")

            sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Score ↓", "Score ↑"], label_visibility="collapsed")

            if sort_by == "Newest":
                filtered_df = filtered_df.sort_values('date', ascending=False)
            elif sort_by == "Oldest":
                filtered_df = filtered_df.sort_values('date', ascending=True)
            elif sort_by == "Score ↓" and 'final_score' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('final_score', ascending=False)
            elif sort_by == "Score ↑" and 'final_score' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('final_score', ascending=True)

            for idx, row in filtered_df.iterrows():
                render_post_card(row, idx)

    with tab_agent:
        render_agent_control()

    with tab_sources:
        st.markdown("### 📡 Content Sources")
        st.markdown("The agent gathers AI/ML content from these platforms:")

        # Use columns for better layout
        source_cols = st.columns(2)
        for idx, (key, config) in enumerate(SOURCE_CONFIG.items()):
            with source_cols[idx % 2]:
                st.markdown(f"""
                    <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem; width: 50px; height: 50px; display: flex; align-items: center;
                                    justify-content: center; border-radius: 10px; background: {config['color']}20;">
                            {config['icon']}
                        </div>
                        <div>
                            <div style="font-weight: 600; font-size: 1.1rem; color: #1e293b;">{config['name']}</div>
                            <div style="color: #64748b; font-size: 0.85rem;">AI-related content</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📰 RSS Feeds")
        st.markdown("""
        **AI Labs:** OpenAI, Anthropic, Google AI, DeepMind, Meta AI, NVIDIA

        **Startups:** Hugging Face, Stability AI, Cohere, Mistral AI, Replicate, Together AI, Groq, Perplexity, LangChain, LlamaIndex

        **Cloud:** AWS ML, Google Cloud AI, Azure AI, Databricks

        **News:** TechCrunch, VentureBeat, Wired, The Verge, MIT Tech Review

        **Research:** BAIR, Stanford AI, Allen AI, EleutherAI
        """)


if __name__ == "__main__":
    main()
