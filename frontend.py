import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import os

# Import your actual logic from the repo
from Train import BasketballImprovementModel
from GameLogsDataset import BasketballPlayerDataset

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Basketball Scout", layout="wide", initial_sidebar_state="collapsed")


# --- LOAD MODEL & DATASET UTILS ---
@st.cache_resource
def load_ai_engine():
    # 1. Load the master tensor data to initialize scalers and shapes
    df_train = pd.read_csv("Data/ml_ready_data.csv")
    master_ds = BasketballPlayerDataset(df=df_train)

    n_leagues = len(master_ds.label_encoders['competition'].classes_)
    n_archetypes = len(master_ds.label_encoders['archetype_cluster'].classes_)

    model = BasketballImprovementModel(cat_dims=[n_leagues, n_archetypes]).to("cpu")

    # 2. Load weights
    weights_path = "Data/basketball_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    return model, master_ds


# --- INITIALIZE ENGINE & DATASETS ---
try:
    ai_model, master_ds = load_ai_engine()

    # Human-readable list for UI mapping
    full_test_df = pd.read_csv("Data/NBA-Test-Result.csv")

    # Unified ML features dataset matching the exact shape of your tensor train pipeline
    ml_features_df = pd.read_csv("Data/ml_ready_data.csv")

    # ==========================================
    # RETIREMENT FILTER: ISOLATE 2024-2025 ACTIVE ROSTER
    # ==========================================
    active_24_25_players = full_test_df[full_test_df['Season'] == '2024-2025']['Player'].unique()
    full_test_df = full_test_df[full_test_df['Player'].isin(active_24_25_players)]

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# --- INITIALIZE STATE ---
# FIXED: Default initialized to the first available index matching our active roster pool
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = int(full_test_df.index[0]) if not full_test_df.empty else 0

# --- UI LOGIC ---
st.title("🏀 AI-Powered Basketball Scout")
search_query = st.text_input("Search players in the test set...", placeholder="e.g., LeBron James")

# Sort and drop duplicates on the human data while retaining index references
display_df = full_test_df.copy()
display_df['original_idx'] = display_df.index  # Lock the alignment pointer

if 'Year' in display_df.columns:
    display_df = display_df.sort_values('Year', ascending=False)
display_df = display_df.drop_duplicates(subset=['Player'], keep='first')

# Apply Search
if search_query:
    display_df = display_df[display_df['Player'].str.contains(search_query, case=False, na=False)]

col_filters, col_list, col_details = st.columns([1.1, 1.2, 2.7], gap="medium")

# ==========================================
# COLUMN 1: INTUITIVE SCOUTING FILTERS (PER GAME)
# ==========================================
with col_filters:
    st.subheader("Scouting Filters")

    # Demographics & League Choices
    available_leagues = display_df['League'].dropna().unique()
    selected_leagues = st.multiselect("Leagues", available_leagues, default=available_leagues)

    min_age = int(display_df['age'].min()) if pd.notna(display_df['age'].min()) else 18
    max_age = int(display_df['age'].max()) if pd.notna(display_df['age'].max()) else 40
    age_range = st.slider("Age", min_age, max_age, (min_age, max_age))

    # Style Profile Dropdown
    archetype_options = ["All Archetypes"] + list(master_ds.label_encoders['archetype_cluster'].classes_)
    selected_archetype = st.selectbox("Player Archetype", archetype_options)

    st.markdown("---")

    # FIXED: Sliders changed to traditional, real-world Per-Game targets instead of fractional Per-Min numbers
    st.markdown("**Minimum Per-Game Targets**")
    min_pts_pg = st.slider("Points / Game", 0.0, 35.0, 0.0, step=1.0)
    min_ast_pg = st.slider("Assists / Game", 0.0, 12.0, 0.0, step=0.5)
    min_reb_pg = st.slider("Rebounds / Game", 0.0, 15.0, 0.0, step=0.5)
    min_stl_pg = st.slider("Steals / Game", 0.0, 3.0, 0.0, step=0.1)
    min_blk_pg = st.slider("Blocks / Game", 0.0, 4.0, 0.0, step=0.1)

    # Initial geographical and demographic slice
    filtered_df = display_df[
        (display_df['League'].isin(selected_leagues)) &
        (display_df['age'] >= age_range[0]) & (display_df['age'] <= age_range[1])
        ]

    # Advanced statistical filtering matrix pass
    # Advanced statistical filtering matrix pass
    # Advanced statistical filtering matrix pass
    # Advanced statistical filtering matrix pass
    valid_indices = []
    for idx in filtered_df['original_idx']:
        # Pull the row directly from your human-readable table
        ui_row = full_test_df.loc[idx]

        # Get total Games Played to calculate traditional averages
        gp = ui_row.get('GP', 1)
        if gp <= 0: gp = 1

        # THE CORRECT MATHEMATICAL CONVERSION:
        # Divide total season statistics by total games played
        pg_pts = ui_row.get('PTS', 0) / gp
        pg_ast = ui_row.get('AST', 0) / gp
        pg_reb = ui_row.get('REB', 0) / gp
        pg_stl = ui_row.get('STL', 0) / gp
        pg_blk = ui_row.get('BLK', 0) / gp

        # Verify if the player's true per-game average passes your slider thresholds
        stat_check = (
                pg_pts >= min_pts_pg and
                pg_ast >= min_ast_pg and
                pg_reb >= min_reb_pg and
                pg_stl >= min_stl_pg and
                pg_blk >= min_blk_pg
        )

        # Verify archetype cluster constraint
        arch_check = True
        if selected_archetype != "All Archetypes":
            # Map column name to 'Archetype' to match the uploaded file
            arch_check = (ui_row.get('Archetype') == selected_archetype)

        if stat_check and arch_check:
            valid_indices.append(idx)

    # Filter the display pool down to the true verified indices
    filtered_df = filtered_df[filtered_df['original_idx'].isin(valid_indices)]
# ==========================================
# BATCH INFERENCE & SYSTEM AUTOMATIC RANKING
# ==========================================
if not filtered_df.empty:
    ai_scores = []
    for _, row in filtered_df.iterrows():
        idx = int(row['original_idx'])
        p_ml_row = ml_features_df.loc[[idx]].copy() # FIXED: .loc matching absolute indexing

        single_player_ds = BasketballPlayerDataset(
            df=p_ml_row,
            scaler=master_ds.scaler,
            label_encoders=master_ds.label_encoders
        )
        x_num, x_cat, _ = single_player_ds[0]

        with torch.no_grad():
            logits = ai_model(x_num.unsqueeze(0), x_cat.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
        ai_scores.append(prob)

    filtered_df['ai_score'] = ai_scores
    filtered_df = filtered_df.sort_values(by='ai_score', ascending=False)

# ==========================================
# COLUMN 2: AUTOMATICALLY RANKED RESULT POOL
# ==========================================
with col_list:
    st.subheader(f"Results ({len(filtered_df)})")
    with st.container(height=720):
        if filtered_df.empty:
            st.info("No prospects match your criteria.")
        else:
            for _, row in filtered_df.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([2.5, 1.5])

                    c1.markdown(f"**{row['Player']}**")
                    c2.markdown(f"`AI: {row['ai_score'] * 100:.1f}%`")

                    safe_age = int(row['age']) if pd.notna(row['age']) else "N/A"
                    c1.caption(f"{row['Team']} | Age: {safe_age}")

                    if st.button("Analyze", key=f"btn_{row['original_idx']}", use_container_width=True):
                        st.session_state.selected_index = int(row['original_idx'])
                        st.rerun()

# ==========================================
# COLUMN 3: LIVE ANALYTICS EXPANSION PANEL
# ==========================================
with col_details:
    # FIXED: Query the frozen data matrices using .loc key mappings directly
    if 'selected_index' in st.session_state and st.session_state.selected_index in ml_features_df.index:
        p_ui_data = full_test_df.loc[st.session_state.selected_index]
        p_ml_row = ml_features_df.loc[[st.session_state.selected_index]].copy()

        single_player_ds = BasketballPlayerDataset(
            df=p_ml_row,
            scaler=master_ds.scaler,
            label_encoders=master_ds.label_encoders
        )
        x_num, x_cat, _ = single_player_ds[0]

        with torch.no_grad():
            logits = ai_model(x_num.unsqueeze(0), x_cat.unsqueeze(0))
            prob = torch.sigmoid(logits).item()

        player_age = p_ui_data['age']

        if prob >= 0.40:
            scout_badge = "✅ PROJECTION: IMPROVEMENT LIKELY"
            badge_type = "success"
        elif prob >= 0.28 and player_age <= 23:
            scout_badge = "⚠️ PROJECTION: HIGH-VARIANCE PROSPECT (Upside Swing)"
            badge_type = "warning"
        elif prob >= 0.32 and 24 <= player_age <= 28:
            scout_badge = "➡️ PROJECTION: PRIME MAINTENANCE"
            badge_type = "info"
        else:
            scout_badge = "❌ PROJECTION: PLATEAU OR DECLINE"
            badge_type = "error"

        st.markdown(f"# {p_ui_data['Player']}")
        st.caption(
            f"Season: {p_ui_data['Season']} | Team: {p_ui_data['Team']} | Age: {int(player_age)} | Archetype: {p_ml_row['archetype_cluster'].values[0]}")

        score_col, result_col = st.columns(2)
        score_col.metric("AI Confidence Score", f"{prob * 100:.1f}%")

        if badge_type == "success":
            result_col.success(scout_badge)
        elif badge_type == "warning":
            result_col.warning(scout_badge)
        elif badge_type == "info":
            result_col.info(scout_badge)
        else:
            result_col.error(scout_badge)

        st.divider()

        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("**Efficiency vs Career Average**")
            metrics = ["Current EFF", "Career Avg", "Trend"]
            vals = [
                p_ml_row.get('cluster_eff').values[0],
                p_ml_row.get('career_eff').values[0],
                p_ml_row.get('trend_eff').values[0]
            ]

            fig_bar = go.Figure(go.Bar(x=metrics, y=vals, marker_color='#E97451'))
            fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("**Advanced Metrics**")
            adv_c1, adv_c2, adv_c3 = st.columns(3)

            ts_pct = p_ml_row.get('ts_pct').values[0]
            trend_eff = p_ml_row.get('trend_eff').values[0]

            adv_c1.metric("True Shooting", f"{ts_pct:.1f}%")
            adv_c2.metric("EFF Trend", f"{trend_eff:.2f}")

            if 'minutes_played' in p_ml_row.columns:
                adv_c3.metric("Minutes Played", int(p_ml_row.get('minutes_played').values[0]))
            else:
                adv_c3.metric("Age Baseline", int(player_age))

        with chart_c2:
            st.markdown("**Archetype Radar**")
            categories = ['PTS/Min', 'AST/Min', 'REB/Min', 'STL/Min', 'BLK/Min']
            stats = [
                p_ml_row.get('points_per_min').values[0],
                p_ml_row.get('assists_per_min').values[0],
                p_ml_row.get('tot_reb_per_min').values[0],
                p_ml_row.get('steals_per_min').values[0],
                p_ml_row.get('blocks_per_min').values[0]
            ]

            fig_radar = go.Figure(go.Scatterpolar(r=stats, theta=categories, fill='toself', line_color='#1f77b4'))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False)),
                height=320,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("Select a player from the list to view their AI Scouting Report.")