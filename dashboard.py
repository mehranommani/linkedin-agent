"""
LinkedIn Content Dashboard
==========================
A beautiful, modern dashboard to review, manage, and publish
your AI-generated LinkedIn posts.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

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
# CUSTOM CSS FOR MODERN UI
# ============================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }

    /* Post card */
    .post-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }

    /* LinkedIn preview */
    .linkedin-preview {
        background: #1a1a1a;
        color: #e0e0e0;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .score-high { background: #d4edda; color: #155724; }
    .score-medium { background: #fff3cd; color: #856404; }
    .score-low { background: #f8d7da; color: #721c24; }

    /* Trending badge */
    .trending-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Copy button */
    .copy-btn {
        background: #0a66c2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        font-weight: 600;
    }

    /* Character count */
    .char-count {
        font-size: 0.8rem;
        color: #666;
    }
    .char-optimal { color: #28a745; }
    .char-warning { color: #ffc107; }
    .char-danger { color: #dc3545; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }

    /* Used post styling */
    .used-badge {
        background: #6c757d;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .used-post {
        opacity: 0.6;
        border-left: 4px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "/Users/mehran/Desktop/Linkedin Agent/linkedin_content_bank.csv"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_data_uncached():
    """Load data without caching (for updates)."""
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(CSV_FILE)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Calculate character count
        if 'linkedin_post' in df.columns:
            df['char_count'] = df['linkedin_post'].str.len()

        # Add 'used' column if it doesn't exist
        if 'used' not in df.columns:
            df['used'] = False

        # Ensure 'used' is boolean
        df['used'] = df['used'].fillna(False).astype(bool)

        return df.sort_values(by='date', ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_data():
    """Load and prepare the data (cached)."""
    return load_data_uncached()


def mark_post_used(original_link: str, used: bool = True):
    """Mark a post as used/unused and save to CSV."""
    try:
        df = pd.read_csv(CSV_FILE)

        # Add 'used' column if it doesn't exist
        if 'used' not in df.columns:
            df['used'] = False

        # Update the specific post
        df.loc[df['original_link'] == original_link, 'used'] = used

        # Save back to CSV
        df.to_csv(CSV_FILE, index=False)

        # Clear cache to reload data
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Error updating post: {e}")
        return False


def get_char_status(count: int) -> tuple:
    """Get character count status and color."""
    if 1400 <= count <= 2100:
        return "Optimal", "char-optimal"
    elif 1000 <= count < 1400 or 2100 < count <= 2500:
        return "Okay", "char-warning"
    else:
        return "Review", "char-danger"


def get_score_class(score: float) -> str:
    """Get score badge class."""
    if score >= 7.0:
        return "score-high"
    elif score >= 5.0:
        return "score-medium"
    return "score-low"


def render_metrics(df: pd.DataFrame):
    """Render the metrics row."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            label="📝 Total Posts",
            value=len(df),
            delta=f"+{len(df[df['date'] == df['date'].max()])}" if len(df) > 0 else None
        )

    with col2:
        # Unused posts count
        if 'used' in df.columns:
            unused = len(df[~df['used']])
            st.metric(
                label="📋 Unused",
                value=unused,
                delta=f"{unused/len(df)*100:.0f}% available" if len(df) > 0 else None
            )
        else:
            st.metric(label="📋 Unused", value=len(df))

    with col3:
        if 'final_score' in df.columns:
            avg_score = df['final_score'].mean()
            st.metric(
                label="⭐ Avg Score",
                value=f"{avg_score:.1f}",
                delta="Good" if avg_score >= 7 else "Improve"
            )
        else:
            st.metric(label="⭐ Avg Score", value="N/A")

    with col4:
        if 'trending_boost' in df.columns:
            trending = len(df[df['trending_boost'] > 0])
            st.metric(
                label="🔥 Trending",
                value=trending,
                delta=f"{trending/len(df)*100:.0f}%" if len(df) > 0 else None
            )
        else:
            st.metric(label="🔥 Trending", value="N/A")

    with col5:
        if 'char_count' in df.columns:
            optimal = len(df[(df['char_count'] >= 1400) & (df['char_count'] <= 2100)])
            st.metric(
                label="📏 Optimal Length",
                value=f"{optimal}/{len(df)}",
                delta=f"{optimal/len(df)*100:.0f}%" if len(df) > 0 else None
            )
        else:
            st.metric(label="📏 Optimal Length", value="N/A")

    with col6:
        if len(df) > 0:
            latest = df['date'].max().strftime('%b %d')
            st.metric(label="📅 Latest", value=latest)
        else:
            st.metric(label="📅 Latest", value="N/A")


def render_post_card(row: pd.Series, index: int):
    """Render a single post card."""
    is_used = row.get('used', False)

    with st.container():
        # Header row with title, badges, and used toggle
        header_col1, header_col2, header_col3 = st.columns([3, 1, 0.5])

        with header_col1:
            title_prefix = "~~" if is_used else ""
            title_suffix = "~~" if is_used else ""
            st.markdown(f"### {title_prefix}{row['title'][:80]}{'...' if len(row['title']) > 80 else ''}{title_suffix}")

        with header_col2:
            badges = ""
            if is_used:
                badges += '<span class="used-badge">✓ USED</span> '
            if 'final_score' in row and pd.notna(row.get('final_score')):
                score_class = get_score_class(row['final_score'])
                badges += f'<span class="score-badge {score_class}">Score: {row["final_score"]:.1f}</span> '
            if 'trending_boost' in row and row.get('trending_boost', 0) > 0:
                badges += '<span class="trending-badge">🔥 TRENDING</span>'
            st.markdown(badges, unsafe_allow_html=True)

        with header_col3:
            # Toggle used status button
            btn_label = "↩️" if is_used else "✓"
            btn_help = "Mark as unused" if is_used else "Mark as used"
            if st.button(btn_label, key=f"toggle_used_{index}", help=btn_help):
                mark_post_used(row['original_link'], not is_used)
                st.rerun()

        # Meta info
        date_str = row['date'].strftime('%B %d, %Y') if pd.notna(row['date']) else 'Unknown'
        st.caption(f"📅 {date_str} • 🔗 [Original Source]({row['original_link']})")

        # Main content area
        col1, col2 = st.columns([1, 2])

        # Left: Image and stats
        with col1:
            # Image
            img_url = row.get('image_url', '')
            if pd.notna(img_url) and str(img_url).startswith('http'):
                st.image(img_url, use_container_width=True)
            else:
                st.info("📷 No image available")

            # Stats
            st.markdown("---")
            st.markdown("**📊 Post Analytics**")

            char_count = len(str(row.get('linkedin_post', '')))
            status, status_class = get_char_status(char_count)
            st.markdown(f"""
            <div style="font-size: 0.9rem;">
                <p><strong>Characters:</strong> <span class="{status_class}">{char_count:,} ({status})</span></p>
            </div>
            """, unsafe_allow_html=True)

            if 'relevance_score' in row and pd.notna(row.get('relevance_score')):
                st.markdown(f"**Relevance:** {row['relevance_score']:.1f}/10")
            if 'quality_score' in row and pd.notna(row.get('quality_score')):
                st.markdown(f"**Quality:** {row['quality_score']:.1f}/10")

        # Right: Post content
        with col2:
            post_text = str(row.get('linkedin_post', ''))

            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 Copy Text", "👀 LinkedIn Preview", "✏️ Edit"])

            with tab1:
                st.text_area(
                    "Click to select all, then copy (Cmd+C / Ctrl+C)",
                    value=post_text,
                    height=350,
                    key=f"copy_{index}",
                    label_visibility="collapsed"
                )
                st.caption("💡 Tip: Click inside the text area and use Cmd+A (Mac) or Ctrl+A (Windows) to select all")

            with tab2:
                st.markdown(f"""
                <div class="linkedin-preview">
                    {post_text}
                </div>
                """, unsafe_allow_html=True)

            with tab3:
                edited = st.text_area(
                    "Edit post content",
                    value=post_text,
                    height=350,
                    key=f"edit_{index}",
                    label_visibility="collapsed"
                )
                if edited != post_text:
                    st.warning("⚠️ Changes are preview only. Manual save not yet implemented.")
                    new_count = len(edited)
                    new_status, new_class = get_char_status(new_count)
                    st.markdown(f"New character count: **{new_count}** ({new_status})")

        st.markdown("---")


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return filtered dataframe."""

    st.sidebar.markdown("## 🎛️ Filters")

    # Search
    search = st.sidebar.text_input("🔍 Search posts", placeholder="Enter keywords...")

    # Date filter
    st.sidebar.markdown("### 📅 Date Range")
    if len(df) > 0:
        dates = df['date'].dt.date.unique()
        date_options = ["All Dates"] + sorted([str(d) for d in dates], reverse=True)
        selected_date = st.sidebar.selectbox("Select date", date_options)
    else:
        selected_date = "All Dates"

    # Score filter
    if 'final_score' in df.columns and len(df) > 0:
        st.sidebar.markdown("### ⭐ Score Range")
        score_min = float(df['final_score'].min())
        score_max = float(df['final_score'].max())
        # Ensure reasonable range
        score_max = max(score_max, 15.0)  # Allow for combined scores > 10
        min_score, max_score = st.sidebar.slider(
            "Filter by score",
            min_value=0.0,
            max_value=score_max,
            value=(0.0, score_max),  # Default to show ALL posts
            step=0.5
        )
    else:
        min_score, max_score = 0.0, 20.0

    # Trending filter
    if 'trending_boost' in df.columns:
        st.sidebar.markdown("### 🔥 Trending")
        trending_only = st.sidebar.checkbox("Show trending only")
    else:
        trending_only = False

    # Used filter
    st.sidebar.markdown("### ✓ Used Status")
    used_filter = st.sidebar.radio(
        "Filter by status",
        ["Unused First", "All Posts", "Unused Only", "Used Only"]
    )

    # Character count filter
    st.sidebar.markdown("### 📏 Length")
    length_filter = st.sidebar.radio(
        "Filter by length",
        ["All", "Optimal (1400-2100)", "Too Short (<1400)", "Too Long (>2100)"]
    )

    # Apply filters
    filtered = df.copy()

    if search:
        mask = (
            filtered['title'].str.contains(search, case=False, na=False) |
            filtered['linkedin_post'].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    if selected_date != "All Dates":
        filtered = filtered[filtered['date'].dt.date == pd.to_datetime(selected_date).date()]

    if 'final_score' in filtered.columns:
        filtered = filtered[
            (filtered['final_score'] >= min_score) &
            (filtered['final_score'] <= max_score)
        ]

    if trending_only and 'trending_boost' in filtered.columns:
        filtered = filtered[filtered['trending_boost'] > 0]

    if 'char_count' in filtered.columns:
        if length_filter == "Optimal (1400-2100)":
            filtered = filtered[(filtered['char_count'] >= 1400) & (filtered['char_count'] <= 2100)]
        elif length_filter == "Too Short (<1400)":
            filtered = filtered[filtered['char_count'] < 1400]
        elif length_filter == "Too Long (>2100)":
            filtered = filtered[filtered['char_count'] > 2100]

    # Apply used filter
    if 'used' in filtered.columns:
        if used_filter == "Unused Only":
            filtered = filtered[~filtered['used']]
        elif used_filter == "Used Only":
            filtered = filtered[filtered['used']]
        elif used_filter == "Unused First":
            # Sort unused posts first (will be applied with other sorting later)
            filtered['_sort_used'] = filtered['used'].astype(int)

    # Stats
    st.sidebar.markdown("---")
    used_count = len(df[df['used'] == True]) if 'used' in df.columns else 0
    unused_count = len(df) - used_count
    st.sidebar.markdown(f"**Showing {len(filtered)} of {len(df)} posts**")
    st.sidebar.markdown(f"📊 {unused_count} unused • {used_count} used")

    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Actions")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("📥 Export to CSV"):
        csv = filtered.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"linkedin_posts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    return filtered


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="margin: 0;">💼 LinkedIn Content Studio</h1>
        <p style="color: #666; margin-top: 0.5rem;">Review, refine, and publish your AI-generated posts</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()

    if df.empty:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 12px;">
            <h2>📭 No Posts Yet</h2>
            <p>Run the agent to generate your first LinkedIn posts:</p>
            <code style="background: #e9ecef; padding: 0.5rem 1rem; border-radius: 4px;">
                python agent_local.py
            </code>
        </div>
        """, unsafe_allow_html=True)
        return

    # Sidebar filters
    filtered_df = render_sidebar(df)

    # Metrics row
    st.markdown("### 📊 Overview")
    render_metrics(df)

    st.markdown("---")

    # Posts
    if len(filtered_df) == 0:
        st.warning("No posts match your filters. Try adjusting the criteria.")
        return

    st.markdown(f"### 📝 Posts ({len(filtered_df)})")

    # Sort options
    sort_col1, sort_col2 = st.columns([1, 4])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (Newest)", "Date (Oldest)", "Score (High)", "Score (Low)", "Length"],
            label_visibility="collapsed"
        )

    # Apply sorting (with used posts pushed to bottom if "Unused First" is selected)
    has_sort_used = '_sort_used' in filtered_df.columns

    if sort_by == "Date (Newest)":
        sort_cols = ['_sort_used', 'date'] if has_sort_used else ['date']
        sort_asc = [True, False] if has_sort_used else [False]
        filtered_df = filtered_df.sort_values(sort_cols, ascending=sort_asc)
    elif sort_by == "Date (Oldest)":
        sort_cols = ['_sort_used', 'date'] if has_sort_used else ['date']
        sort_asc = [True, True] if has_sort_used else [True]
        filtered_df = filtered_df.sort_values(sort_cols, ascending=sort_asc)
    elif sort_by == "Score (High)" and 'final_score' in filtered_df.columns:
        sort_cols = ['_sort_used', 'final_score'] if has_sort_used else ['final_score']
        sort_asc = [True, False] if has_sort_used else [False]
        filtered_df = filtered_df.sort_values(sort_cols, ascending=sort_asc)
    elif sort_by == "Score (Low)" and 'final_score' in filtered_df.columns:
        sort_cols = ['_sort_used', 'final_score'] if has_sort_used else ['final_score']
        sort_asc = [True, True] if has_sort_used else [True]
        filtered_df = filtered_df.sort_values(sort_cols, ascending=sort_asc)
    elif sort_by == "Length" and 'char_count' in filtered_df.columns:
        sort_cols = ['_sort_used', 'char_count'] if has_sort_used else ['char_count']
        sort_asc = [True, False] if has_sort_used else [False]
        filtered_df = filtered_df.sort_values(sort_cols, ascending=sort_asc)

    # Clean up temp column
    if has_sort_used:
        filtered_df = filtered_df.drop(columns=['_sort_used'])

    # Render posts
    for idx, row in filtered_df.iterrows():
        render_post_card(row, idx)


if __name__ == "__main__":
    main()
