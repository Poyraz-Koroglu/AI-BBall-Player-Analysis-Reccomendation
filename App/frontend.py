import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Basketball Analytics", layout="wide", initial_sidebar_state="collapsed")

# --- INITIALIZE STATE ---
# We use session state to remember which player is currently selected for the details view
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = 'Nikola Jokic'


# --- MOCK DATA LOAD ---
@st.cache_data
def load_mock_data():
    data = {
        'Name': ['Stephen Curry', 'LeBron James', 'Nikola Jokic', 'Luka Doncic', 'Anthony Edwards'],
        'Team': ['GSW', 'LAL', 'DEN', 'DAL', 'MIN'],
        'Pos': ['PG', 'SF', 'C', 'PG', 'SG'],
        'Age': [36, 39, 29, 25, 22],
        'Score': [95, 96, 98, 97, 92],
        'PER': [21.3, 23.1, 31.0, 28.5, 19.8],
        'WS': [8.4, 7.9, 13.5, 10.1, 6.5],
        'USG': [30.1, 28.5, 29.3, 36.0, 31.5],
        'PPG': [26.4, 25.7, 26.4, 33.9, 25.9],
        'RPG': [4.5, 7.3, 12.4, 9.2, 5.4],
        'APG': [5.1, 8.3, 9.0, 9.8, 5.1],
        # Radar Percentiles (Scoring, Shooting, Passing, Rebounding, Defense, Efficiency)
        'R_Scoring': [95, 92, 90, 99, 90],
        'R_Shooting': [98, 85, 88, 85, 82],
        'R_Passing': [90, 95, 99, 98, 75],
        'R_Rebounding': [40, 75, 98, 80, 60],
        'R_Defense': [50, 70, 75, 55, 85],
        'R_Efficiency': [92, 90, 99, 90, 80],
        # Historical PPG for last 5 years
        'Hist_Y1': [20.8, 25.3, 19.9, 28.8, 19.3],
        'Hist_Y2': [32.0, 25.0, 26.4, 27.7, 21.3],
        'Hist_Y3': [25.5, 30.3, 27.1, 28.4, 24.6],
        'Hist_Y4': [29.4, 28.9, 24.5, 32.4, 25.9],
        'Hist_Y5': [26.4, 25.7, 26.4, 33.9, 25.9],
    }
    return pd.DataFrame(data)


df = load_mock_data()

# --- TOP SEARCH BAR ---
st.title("🏀 Basketball Analytics")
search_query = st.text_input("Search players...", placeholder="e.g., Stephen Curry")

# Apply search filter
if search_query:
    df = df[df['Name'].str.contains(search_query, case=False)]

# --- LAYOUT: THREE COLUMNS ---
# Adjusting ratios: Left (Filters), Middle (List), Right (Analytics)
col_filters, col_list, col_details = st.columns([1, 1.2, 2.8], gap="large")

# ==========================================
# COLUMN 1: FILTERS
# ==========================================
with col_filters:
    st.subheader("Filters")

    # Position Filter
    st.markdown("**Positions**")
    selected_pos = st.multiselect("Select Positions", ['PG', 'SG', 'SF', 'PF', 'C'],
                                  default=['PG', 'SG', 'SF', 'PF', 'C'])

    # Age Filter
    st.markdown("**Age Range**")
    age_range = st.slider("Select Age", 18, 40, (18, 40), label_visibility="collapsed")

    # Player Score Filter
    st.markdown("**Player Score**")
    score_range = st.slider("Select Score", 0, 100, (80, 100), label_visibility="collapsed")

    # Apply Filters to DataFrame
    filtered_df = df[
        (df['Pos'].isin(selected_pos)) &
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Score'] >= score_range[0]) & (df['Score'] <= score_range[1])
        ]

# ==========================================
# COLUMN 2: RECOMMENDED PLAYERS LIST
# ==========================================
with col_list:
    st.subheader("Recommended")

    if filtered_df.empty:
        st.warning("No players match your filters.")
    else:
        # Create a scrollable container for the player cards
        with st.container(height=700):
            for _, row in filtered_df.iterrows():
                # Highlight the card if it's the currently selected player
                is_selected = row['Name'] == st.session_state.selected_player
                border_color = "primary" if is_selected else "secondary"

                # Build the card
                with st.container(border=True):
                    card_col1, card_col2 = st.columns([3, 1])
                    with card_col1:
                        st.markdown(f"**{row['Name']}**")
                        st.caption(f"{row['Team']}, {row['Pos']} | Age: {row['Age']}")
                    with card_col2:
                        st.markdown(f"### {row['Score']}")

                    # The button that updates the right column
                    if st.button("Analyze", key=f"btn_{row['Name']}", use_container_width=True):
                        st.session_state.selected_player = row['Name']
                        st.rerun()  # Force the app to refresh with the new selection

# ==========================================
# COLUMN 3: DETAILED ANALYTICS
# ==========================================
with col_details:
    # Get the data for the currently selected player
    # Fallback to the first player in the filtered list if the selected player was filtered out
    if st.session_state.selected_player not in filtered_df['Name'].values and not filtered_df.empty:
        player_data = filtered_df.iloc[0]
    elif filtered_df.empty:
        player_data = None
    else:
        player_data = filtered_df[filtered_df['Name'] == st.session_state.selected_player].iloc[0]

    if player_data is not None:
        # Header Area
        head_c1, head_c2 = st.columns([2, 1])
        with head_c1:
            st.markdown(f"# {player_data['Name']}")
            st.markdown(
                f"**{player_data['Team']}** | Position: **{player_data['Pos']}** | Age: **{player_data['Age']}**")
        with head_c2:
            st.metric("Hoops Performance Score", f"{player_data['Score']}/100")

        st.divider()

        # Top Row of Charts (Line Chart & Radar Chart)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**PPG - Last 5 Seasons**")
            hist_data = [player_data['Hist_Y1'], player_data['Hist_Y2'], player_data['Hist_Y3'], player_data['Hist_Y4'],
                         player_data['Hist_Y5']]
            years = ['2020', '2021', '2022', '2023', '2024']

            fig_line = go.Figure(
                go.Scatter(x=years, y=hist_data, mode='lines+markers', fill='tozeroy', line_color='#E97451'))
            fig_line.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=250)
            st.plotly_chart(fig_line, use_container_width=True)

            # Advanced Stats under the line chart
            st.markdown("**Advanced Stats**")
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            adv_c1.metric("PER", player_data['PER'])
            adv_c2.metric("Win Shares", player_data['WS'])
            adv_c3.metric("Usage %", f"{player_data['USG']}%")

        with chart_col2:
            st.markdown("**Performance Radar**")
            categories = ['Scoring', 'Shooting', 'Passing', 'Rebounding', 'Defense', 'Efficiency']
            percentiles = [
                player_data['R_Scoring'], player_data['R_Shooting'], player_data['R_Passing'],
                player_data['R_Rebounding'], player_data['R_Defense'], player_data['R_Efficiency']
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=percentiles + [percentiles[0]],
                theta=categories + [categories[0]],
                fill='toself', line_color='#E97451'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[50, 50, 50, 50, 50, 50, 50],
                theta=categories + [categories[0]],
                fill='none', line_color='gray', line_dash='dash', name='Avg'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 100])),
                showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=280
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # Bottom Row: Bar Chart vs Position Averages
        st.markdown(f"**Comparing to {player_data['Pos']} Averages**")

        # Hardcoded mock averages for demonstration
        pos_avgs = {'PG': [18, 4, 6], 'SG': [17, 4.5, 4], 'SF': [16, 6, 3], 'PF': [15, 8, 2], 'C': [14, 10, 2]}
        avg_data = pos_avgs.get(player_data['Pos'], [15, 5, 5])
        player_stats = [player_data['PPG'], player_data['RPG'], player_data['APG']]
        stat_names = ['PPG', 'RPG', 'APG']

        fig_bar = go.Figure(data=[
            go.Bar(name=player_data['Name'], x=stat_names, y=player_stats, marker_color='#1f77b4'),
            go.Bar(name=f"{player_data['Pos']} Avg", x=stat_names, y=avg_data, marker_color='#ff7f0e')
        ])
        fig_bar.update_layout(barmode='group', margin=dict(l=20, r=20, t=20, b=20), height=250)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Select a player from the list to view their analytics.")