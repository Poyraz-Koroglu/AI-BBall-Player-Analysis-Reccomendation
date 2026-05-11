import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
import os
# Import your actual logic from the repo
from Dataset import BasketballPlayerDataset
from Model import ImprovementPredictor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Basketball Scout", layout="wide", initial_sidebar_state="collapsed")


# --- LOAD MODEL & DATASET UTILS ---
@st.cache_resource
def load_ai_engine():
    # 1. Load original training data to reconstruct the exact model shape and scalers
    # This ensures team IDs and stat scaling match the training environment
    df_train = pd.read_csv("Data/final_training_data_cumulative.csv")

    # Initialize the dataset once to harvest its scaler and label encoders
    master_ds = BasketballPlayerDataset(df=df_train, target_col='Improved')

    # 2. Rebuild the Model Architecture
    cat_cardinalities = [len(master_ds.label_encoders[col].classes_) + 1 for col in master_ds.categorical_cols]
    model = ImprovementPredictor(
        num_numerical_features=len(master_ds.numerical_cols),
        categorical_cardinalities=cat_cardinalities,
        hidden_units=[256, 128, 64]
    )

    # 3. Load the saved weights
    if os.path.exists("basketball_model_best.pth"):
        model.load_state_dict(torch.load("basketball_model_best.pth", map_location="cpu"))
    model.eval()

    return model, master_ds


# --- INITIALIZE ENGINE ---
try:
    ai_model, master_ds = load_ai_engine()
    # Load your actual test dataset for the player list
    full_test_df = pd.read_csv("Data/NBA-Test-Result.csv")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# --- INITIALIZE STATE ---
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = full_test_df['Player'].iloc[0]

# --- UI LOGIC ---
st.title("🏀 AI-Powered Basketball Scout")
search_query = st.text_input("Search players in the test set...", placeholder="e.g., LeBron James")

# 1. FIX: Sort by Year (newest first) and drop duplicates so each player only gets ONE card
display_df = full_test_df.copy()
if 'Year' in display_df.columns:
    display_df = display_df.sort_values('Year', ascending=False)
display_df = display_df.drop_duplicates(subset=['Player'], keep='first')

# Apply Search
if search_query:
    display_df = display_df[display_df['Player'].str.contains(search_query, case=False, na=False)]

col_filters, col_list, col_details = st.columns([1, 1.2, 2.8], gap="large")

# ==========================================
# COLUMN 1: FILTERS (Using Actual Data Columns)
# ==========================================
with col_filters:
    st.subheader("Scouting Filters")

    # Safely handle leagues and ages
    available_leagues = display_df['League'].dropna().unique()
    selected_leagues = st.multiselect("Leagues", available_leagues, default=available_leagues)

    min_age = int(display_df['age'].min()) if pd.notna(display_df['age'].min()) else 18
    max_age = int(display_df['age'].max()) if pd.notna(display_df['age'].max()) else 40
    age_range = st.slider("Age", min_age, max_age, (min_age, max_age))

    filtered_df = display_df[
        (display_df['League'].isin(selected_leagues)) &
        (display_df['age'] >= age_range[0]) & (display_df['age'] <= age_range[1])
        ]

# ==========================================
# COLUMN 2: PLAYER LIST
# ==========================================
with col_list:
    st.subheader("Results")
    with st.container(height=700):
        for _, row in filtered_df.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{row['Player']}**")

                # 2. FIX: Safely display age in case of missing data (which breaks buttons)
                safe_age = int(row['age']) if pd.notna(row['age']) else "N/A"
                c1.caption(f"{row['Team']} | Age: {safe_age}")

                # 3. FIX: Simplified, unique button key
                if st.button("Analyze", key=f"btn_{row['Player']}", use_container_width=True):
                    st.session_state.selected_player = row['Player']
                    st.rerun()

# ==========================================
# COLUMN 3: LIVE AI INFERENCE & ANALYTICS
# ==========================================
with col_details:
    # 1. Grab the exact row for the selected player
    player_row = filtered_df[filtered_df['Player'] == st.session_state.selected_player]

    if not player_row.empty:
        p_data = player_row.iloc[0]

        # --- AI SCOUT INFERENCE ---
        # Wrap the single player row in the Dataset class to ensure correct preprocessing
        single_player_ds = BasketballPlayerDataset(
            df=player_row,
            scaler=master_ds.scaler,
            label_encoders=master_ds.label_encoders
        )
        x_num, x_cat = single_player_ds[0]

        # Run the neural network
        with torch.no_grad():
            logits = ai_model(x_num.unsqueeze(0), x_cat.unsqueeze(0))
            prob = torch.sigmoid(logits).item()

        # --- DYNAMIC AGE-BASED SCOUTING TIERS ---
        player_age = p_data['age']

        # 1. The Sure-Fire Improvers (Any age, high confidence)
        if prob >= 0.55:
            scout_badge = "✅ PROJECTION: STRONG IMPROVEMENT LIKELY"
            badge_type = "success"

        # 2. The Young Upside Swings (Under 24, 35%-54% confidence)
        elif prob >= 0.35 and player_age <= 23:
            scout_badge = "⚠️ PROJECTION: HIGH-VARIANCE PROSPECT (Upside Swing)"
            badge_type = "warning"

        # 3. The Prime Maintainers (Age 24-28, borderline confidence)
        elif prob >= 0.40 and 24 <= player_age <= 28:
            scout_badge = "➡️ PROJECTION: PRIME MAINTENANCE"
            badge_type = "info"

        # 4. The Decliners (Everyone else)
        else:
            scout_badge = "❌ PROJECTION: PLATEAU OR DECLINE"
            badge_type = "error"

        # --- HEADER UI ---
        st.markdown(f"# {p_data['Player']}")
        st.caption(f"Season: {p_data['Season']} | Team: {p_data['Team']} | Age: {int(player_age)}")

        score_col, result_col = st.columns(2)
        score_col.metric("AI Confidence Score", f"{prob * 100:.1f}%")

        # Render the dynamic badge based on the tier
        if badge_type == "success":
            result_col.success(scout_badge)
        elif badge_type == "warning":
            result_col.warning(scout_badge)
        elif badge_type == "info":
            result_col.info(scout_badge)
        else:
            result_col.error(scout_badge)

        st.divider()

        # --- VISUALIZATIONS & ADVANCED STATS ---
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("**Efficiency vs Career Average**")
            # Using current EFF, historical average, and Trend from your dataset
            metrics = ["Current EFF", "Career Avg", "Trend"]
            vals = [p_data['EFF_per_min'], p_data['Career_EFF_Avg'], p_data['Trend_EFF']]

            fig_bar = go.Figure(go.Bar(x=metrics, y=vals, marker_color='#E97451'))
            fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

            # Advanced metrics (Replaced PER with Real AI Inputs)
            st.markdown("**Advanced Metrics**")
            adv_c1, adv_c2, adv_c3 = st.columns(3)

            # Use .get() safely in case any columns are named slightly differently
            ts_pct = p_data.get('TS_pct', 0)
            usg_pct = p_data.get('USG_pct', 0)  # Change to a different stat if USG isn't in your CSV

            adv_c1.metric("True Shooting", f"{ts_pct * 100:.1f}%")
            adv_c2.metric("EFF Trend", f"{p_data['Trend_EFF']:.2f}")

            # If you don't have usage, we fall back to something universally useful like minutes
            if 'USG_pct' in p_data:
                adv_c3.metric("Usage %", f"{usg_pct * 100:.1f}%")
            else:
                adv_c3.metric("Career Mins", int(p_data.get('Career_MIN', 0)))

        with chart_c2:
            st.markdown("**Archetype Radar**")
            # Categories based on your advanced feature engineering
            categories = ['PTS/Min', 'AST/Min', 'REB/Min', 'TS%', 'FT Rate']
            stats = [
                p_data.get('PTS_per_min', 0),
                p_data.get('AST_per_min', 0),
                p_data.get('REB_per_min', 0),
                p_data.get('TS_pct', 0),
                p_data.get('FT_Rate', 0)
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