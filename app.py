import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "win"
    elif row["home_score"] < row["away_score"]:
        return "loss"
    else:
        return "draw"

def get_form(team, date, df):
    past_matches = df[df["date"] < date]
    team_matches = past_matches[
        (past_matches["home_team"] == team) | (past_matches["away_team"] == team)
    ]
    last_5 = team_matches.sort_values("date").tail(5)
    points = 0
    for _, match in last_5.iterrows():
        if match["home_team"] == team:
            if match["home_score"] > match["away_score"]:
                points += 3
            elif match["home_score"] == match["away_score"]:
                points += 1
        else:
            if match["away_score"] > match["home_score"]:
                points += 3
            elif match["away_score"] == match["home_score"]:
                points += 1
    return points

# Name corrections: fixture name → dataset name
NAME_MAP = {
    "United States" : "United States",
    "Turkey"        : "Turkey",
    "DR Congo"      : "DR Congo",
    "Ivory Coast"   : "Ivory Coast",
    "Cape Verde"    : "Cape Verde",
    "Iran"          : "IR Iran",
    "Curacao"       : "Curacao",
}

def resolve(team):
    return NAME_MAP.get(team, team)


# ─────────────────────────────────────────────
#  DATA LOADING & MODEL TRAINING
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    data = requests.get(
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    ).text
    df = pd.read_csv(StringIO(data), parse_dates=["date"])
    df = df[df["tournament"] != "Friendly"]
    df = df[df["date"] >= "2000-01-01"]
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"]  = df["away_score"].astype(int)
    df["result"]      = df.apply(get_result, axis=1)
    return df

@st.cache_data
def compute_elo(_df, k=32, base=1500):
    elo = {}
    for _, row in _df.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        h_elo = elo.get(home, base)
        a_elo = elo.get(away, base)
        expected = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        actual = {"win": 1, "draw": 0.5, "loss": 0}[row["result"]]
        elo[home] = h_elo + k * (actual - expected)
        elo[away] = a_elo + k * ((1 - actual) - (1 - expected))
    return elo

@st.cache_data
def build_features(_df, _elo):
    records = []
    for _, row in _df.iterrows():
        h_elo  = _elo.get(row["home_team"], 1500)
        a_elo  = _elo.get(row["away_team"], 1500)
        h_form = get_form(row["home_team"], row["date"], _df)
        a_form = get_form(row["away_team"], row["date"], _df)
        records.append({
            "elo_diff"  : h_elo - a_elo,
            "is_neutral": int(row["neutral"]),
            "home_form" : h_form,
            "away_form" : a_form,
            "form_diff" : h_form - a_form,
            "result"    : row["result"],
        })
    return pd.DataFrame(records)

@st.cache_resource
def train_model(features_df):
    X = features_df[["elo_diff", "is_neutral", "home_form", "away_form", "form_diff"]]
    y = features_df["result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    clf.fit(X_train, y_train)
    return clf

@st.cache_data
def precompute_forms(_df):
    """Precompute form for every team as of World Cup kickoff — avoids slow repeated calls."""
    cutoff = pd.Timestamp("2026-06-01")
    teams  = set(_df["home_team"].unique()) | set(_df["away_team"].unique())
    return {team: get_form(team, cutoff, _df) for team in teams}


# ─────────────────────────────────────────────
#  PREDICTION UTILITIES
# ─────────────────────────────────────────────

def get_probs(team1, team2, elo, clf, team_forms):
    t1_key = resolve(team1)
    t2_key = resolve(team2)
    t1_elo, t2_elo   = elo.get(t1_key, 1500), elo.get(t2_key, 1500)
    t1_form, t2_form = team_forms.get(t1_key, 0), team_forms.get(t2_key, 0)
    X = pd.DataFrame([{
        "elo_diff"  : t1_elo - t2_elo,
        "is_neutral": 1,
        "home_form" : t1_form,
        "away_form" : t2_form,
        "form_diff" : t1_form - t2_form,
    }])
    probs = clf.predict_proba(X)[0]
    return dict(zip(clf.classes_, probs))

def knockout_winner(team1, team2, elo, clf, team_forms):
    """No draws in knockout — split draw probability 50/50."""
    prob_dict = get_probs(team1, team2, elo, clf, team_forms)
    t1_p = prob_dict.get("win", 0)  + prob_dict.get("draw", 0) * 0.5
    t2_p = prob_dict.get("loss", 0) + prob_dict.get("draw", 0) * 0.5
    if t1_p >= t2_p:
        return team1, team2, round(t1_p * 100, 1)   # winner, loser, confidence
    else:
        return team2, team1, round(t2_p * 100, 1)


# ─────────────────────────────────────────────
#  TOURNAMENT SIMULATION
# ─────────────────────────────────────────────

def simulate_group_stage(fixtures, elo, clf, team_forms):
    groups = list("ABCDEFGHIJKL")
    group_results = {}

    for g in groups:
        g_fixtures = [f for f in fixtures if f["group"] == g]
        # preserve order so ranked list is stable
        seen, teams = set(), []
        for f in g_fixtures:
            for t in [f["team1"], f["team2"]]:
                if t not in seen:
                    seen.add(t); teams.append(t)

        pts = {t: 0 for t in teams}

        for match in g_fixtures:
            t1, t2    = match["team1"], match["team2"]
            prob_dict = get_probs(t1, t2, elo, clf, team_forms)
            predicted = max(prob_dict, key=prob_dict.get)
            if predicted == "win":
                pts[t1] += 3
            elif predicted == "draw":
                pts[t1] += 1; pts[t2] += 1
            else:
                pts[t2] += 3

        ranked = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        group_results[g] = {
            "ranked": [t for t, _ in ranked],
            "pts"   : pts,
        }

    return group_results


def assign_third_place(group_results):
    """Rank all 12 third-place teams, take top 8, assign to R32 slots by pool rules."""
    thirds = [
        (g, data["ranked"][2], data["pts"][data["ranked"][2]])
        for g, data in group_results.items()
        if len(data["ranked"]) >= 3
    ]
    thirds_sorted = sorted(thirds, key=lambda x: x[2], reverse=True)
    qualifying    = thirds_sorted[:8]   # top 8 advance

    # each slot can only receive a 3rd-place team from specific groups
    slots = [
        (74, ["A","B","C","D","F"]),
        (77, ["C","D","F","G","H"]),
        (79, ["C","E","F","H","I"]),
        (80, ["E","H","I","J","K"]),
        (81, ["B","E","F","I","J"]),
        (82, ["A","E","H","I","J"]),
        (85, ["E","F","G","I","J"]),
        (87, ["D","E","I","J","L"]),
    ]

    available   = list(qualifying)
    assignments = {}
    for match_num, pools in slots:
        for i, (grp, team, _) in enumerate(available):
            if grp in pools:
                assignments[match_num] = team
                available.pop(i)
                break
        else:                               # fallback: best remaining
            if available:
                _, team, _ = available.pop(0)
                assignments[match_num] = team

    return assignments


def simulate_knockout(group_results, third_assignments, elo, clf, team_forms):
    gr      = group_results
    results = {}   # match_num → {winner, loser, t1, t2, prob}

    def play(match_num, t1, t2):
        w, l, prob = knockout_winner(t1, t2, elo, clf, team_forms)
        results[match_num] = {"winner": w, "loser": l, "t1": t1, "t2": t2, "prob": prob}

    # ── Round of 32 ──────────────────────────
    play(73, gr["A"]["ranked"][1], gr["B"]["ranked"][1])
    play(74, gr["E"]["ranked"][0], third_assignments.get(74, "TBD"))
    play(75, gr["F"]["ranked"][0], gr["C"]["ranked"][1])
    play(76, gr["C"]["ranked"][0], gr["F"]["ranked"][1])
    play(77, gr["I"]["ranked"][0], third_assignments.get(77, "TBD"))
    play(78, gr["E"]["ranked"][1], gr["I"]["ranked"][1])
    play(79, gr["A"]["ranked"][0], third_assignments.get(79, "TBD"))
    play(80, gr["L"]["ranked"][0], third_assignments.get(80, "TBD"))
    play(81, gr["D"]["ranked"][0], third_assignments.get(81, "TBD"))
    play(82, gr["G"]["ranked"][0], third_assignments.get(82, "TBD"))
    play(83, gr["K"]["ranked"][1], gr["L"]["ranked"][1])
    play(84, gr["H"]["ranked"][0], gr["J"]["ranked"][1])
    play(85, gr["B"]["ranked"][0], third_assignments.get(85, "TBD"))
    play(86, gr["J"]["ranked"][0], gr["H"]["ranked"][1])
    play(87, gr["K"]["ranked"][0], third_assignments.get(87, "TBD"))
    play(88, gr["D"]["ranked"][1], gr["G"]["ranked"][1])

    # ── Round of 16 ──────────────────────────
    play(89, results[74]["winner"], results[77]["winner"])
    play(90, results[73]["winner"], results[75]["winner"])
    play(91, results[76]["winner"], results[78]["winner"])
    play(92, results[79]["winner"], results[80]["winner"])
    play(93, results[83]["winner"], results[84]["winner"])
    play(94, results[81]["winner"], results[82]["winner"])
    play(95, results[86]["winner"], results[88]["winner"])
    play(96, results[85]["winner"], results[87]["winner"])

    # ── Quarter-finals ────────────────────────
    play(97,  results[89]["winner"], results[90]["winner"])
    play(98,  results[93]["winner"], results[94]["winner"])
    play(99,  results[91]["winner"], results[92]["winner"])
    play(100, results[95]["winner"], results[96]["winner"])

    # ── Semi-finals ───────────────────────────
    play(101, results[97]["winner"],  results[98]["winner"])
    play(102, results[99]["winner"],  results[100]["winner"])

    # ── Third place ───────────────────────────
    play(103, results[101]["loser"],  results[102]["loser"])

    # ── Final ─────────────────────────────────
    play(104, results[101]["winner"], results[102]["winner"])

    return results

#  FIXTURES (all 72 group stage matches)

FIXTURES = [
    # Group A
    {"date": "Jun 11", "team1": "Mexico",                 "team2": "South Africa",           "group": "A"},
    {"date": "Jun 11", "team1": "South Korea",            "team2": "Czechia",                "group": "A"},
    {"date": "Jun 18", "team1": "Czechia",                "team2": "South Africa",           "group": "A"},
    {"date": "Jun 18", "team1": "Mexico",                 "team2": "South Korea",            "group": "A"},
    {"date": "Jun 24", "team1": "Czechia",                "team2": "Mexico",                 "group": "A"},
    {"date": "Jun 24", "team1": "South Africa",           "team2": "South Korea",            "group": "A"},
    # Group B
    {"date": "Jun 12", "team1": "Canada",                 "team2": "Bosnia and Herzegovina", "group": "B"},
    {"date": "Jun 13", "team1": "Qatar",                  "team2": "Switzerland",            "group": "B"},
    {"date": "Jun 18", "team1": "Switzerland",            "team2": "Bosnia and Herzegovina", "group": "B"},
    {"date": "Jun 18", "team1": "Canada",                 "team2": "Qatar",                  "group": "B"},
    {"date": "Jun 24", "team1": "Switzerland",            "team2": "Canada",                 "group": "B"},
    {"date": "Jun 24", "team1": "Bosnia and Herzegovina", "team2": "Qatar",                  "group": "B"},
    # Group C
    {"date": "Jun 13", "team1": "Haiti",                  "team2": "Scotland",               "group": "C"},
    {"date": "Jun 13", "team1": "Brazil",                 "team2": "Morocco",                "group": "C"},
    {"date": "Jun 19", "team1": "Brazil",                 "team2": "Haiti",                  "group": "C"},
    {"date": "Jun 19", "team1": "Scotland",               "team2": "Morocco",                "group": "C"},
    {"date": "Jun 24", "team1": "Scotland",               "team2": "Brazil",                 "group": "C"},
    {"date": "Jun 24", "team1": "Morocco",                "team2": "Haiti",                  "group": "C"},
    # Group D
    {"date": "Jun 12", "team1": "United States",          "team2": "Paraguay",               "group": "D"},
    {"date": "Jun 13", "team1": "Australia",              "team2": "Turkey",                 "group": "D"},
    {"date": "Jun 19", "team1": "Turkey",                 "team2": "Paraguay",               "group": "D"},
    {"date": "Jun 19", "team1": "United States",          "team2": "Australia",              "group": "D"},
    {"date": "Jun 25", "team1": "Turkey",                 "team2": "United States",          "group": "D"},
    {"date": "Jun 25", "team1": "Paraguay",               "team2": "Australia",              "group": "D"},
    # Group E
    {"date": "Jun 14", "team1": "Ivory Coast",            "team2": "Ecuador",                "group": "E"},
    {"date": "Jun 14", "team1": "Germany",                "team2": "Curacao",                "group": "E"},
    {"date": "Jun 20", "team1": "Germany",                "team2": "Ivory Coast",            "group": "E"},
    {"date": "Jun 20", "team1": "Ecuador",                "team2": "Curacao",                "group": "E"},
    {"date": "Jun 25", "team1": "Curacao",                "team2": "Ivory Coast",            "group": "E"},
    {"date": "Jun 25", "team1": "Ecuador",                "team2": "Germany",                "group": "E"},
    # Group F
    {"date": "Jun 14", "team1": "Netherlands",            "team2": "Japan",                  "group": "F"},
    {"date": "Jun 14", "team1": "Sweden",                 "team2": "Tunisia",                "group": "F"},
    {"date": "Jun 20", "team1": "Netherlands",            "team2": "Sweden",                 "group": "F"},
    {"date": "Jun 20", "team1": "Tunisia",                "team2": "Japan",                  "group": "F"},
    {"date": "Jun 25", "team1": "Japan",                  "team2": "Sweden",                 "group": "F"},
    {"date": "Jun 25", "team1": "Tunisia",                "team2": "Netherlands",            "group": "F"},
    # Group G
    {"date": "Jun 15", "team1": "Iran",                   "team2": "New Zealand",            "group": "G"},
    {"date": "Jun 15", "team1": "Belgium",                "team2": "Egypt",                  "group": "G"},
    {"date": "Jun 21", "team1": "Belgium",                "team2": "Iran",                   "group": "G"},
    {"date": "Jun 21", "team1": "New Zealand",            "team2": "Egypt",                  "group": "G"},
    {"date": "Jun 26", "team1": "Egypt",                  "team2": "Iran",                   "group": "G"},
    {"date": "Jun 26", "team1": "New Zealand",            "team2": "Belgium",                "group": "G"},
    # Group H
    {"date": "Jun 15", "team1": "Saudi Arabia",           "team2": "Uruguay",                "group": "H"},
    {"date": "Jun 15", "team1": "Spain",                  "team2": "Cape Verde",             "group": "H"},
    {"date": "Jun 21", "team1": "Uruguay",                "team2": "Cape Verde",             "group": "H"},
    {"date": "Jun 21", "team1": "Spain",                  "team2": "Saudi Arabia",           "group": "H"},
    {"date": "Jun 26", "team1": "Cape Verde",             "team2": "Saudi Arabia",           "group": "H"},
    {"date": "Jun 26", "team1": "Uruguay",                "team2": "Spain",                  "group": "H"},
    # Group I
    {"date": "Jun 16", "team1": "France",                 "team2": "Senegal",                "group": "I"},
    {"date": "Jun 16", "team1": "Iraq",                   "team2": "Norway",                 "group": "I"},
    {"date": "Jun 22", "team1": "Norway",                 "team2": "Senegal",                "group": "I"},
    {"date": "Jun 22", "team1": "France",                 "team2": "Iraq",                   "group": "I"},
    {"date": "Jun 26", "team1": "Norway",                 "team2": "France",                 "group": "I"},
    {"date": "Jun 26", "team1": "Senegal",                "team2": "Iraq",                   "group": "I"},
    # Group J
    {"date": "Jun 16", "team1": "Argentina",              "team2": "Algeria",                "group": "J"},
    {"date": "Jun 16", "team1": "Austria",                "team2": "Jordan",                 "group": "J"},
    {"date": "Jun 22", "team1": "Argentina",              "team2": "Austria",                "group": "J"},
    {"date": "Jun 22", "team1": "Jordan",                 "team2": "Algeria",                "group": "J"},
    {"date": "Jun 27", "team1": "Algeria",                "team2": "Austria",                "group": "J"},
    {"date": "Jun 27", "team1": "Jordan",                 "team2": "Argentina",              "group": "J"},
    # Group K
    {"date": "Jun 17", "team1": "Portugal",               "team2": "DR Congo",               "group": "K"},
    {"date": "Jun 17", "team1": "Uzbekistan",             "team2": "Colombia",               "group": "K"},
    {"date": "Jun 23", "team1": "Portugal",               "team2": "Uzbekistan",             "group": "K"},
    {"date": "Jun 23", "team1": "Colombia",               "team2": "DR Congo",               "group": "K"},
    {"date": "Jun 27", "team1": "Colombia",               "team2": "Portugal",               "group": "K"},
    {"date": "Jun 27", "team1": "DR Congo",               "team2": "Uzbekistan",             "group": "K"},
    # Group L
    {"date": "Jun 17", "team1": "Ghana",                  "team2": "Panama",                 "group": "L"},
    {"date": "Jun 17", "team1": "England",                "team2": "Croatia",                "group": "L"},
    {"date": "Jun 23", "team1": "England",                "team2": "Ghana",                  "group": "L"},
    {"date": "Jun 23", "team1": "Panama",                 "team2": "Croatia",                "group": "L"},
    {"date": "Jun 27", "team1": "Panama",                 "team2": "England",                "group": "L"},
    {"date": "Jun 27", "team1": "Croatia",                "team2": "Ghana",                  "group": "L"},
]

#  LOAD & TRAIN

with st.spinner("Loading data and training model... (first run ~2 mins)"):
    df         = load_data()
    elo        = compute_elo(df)
    features   = build_features(df, elo)
    clf        = train_model(features)
    team_forms = precompute_forms(df)


#  UI

st.title("⚽ 2026 World Cup Predictor")

tab1, tab2, tab3 = st.tabs(["🔮 Custom Prediction", "📅 Group Stage Schedule", "🏆 Tournament Simulator"])


# Tab 1: Custom Prediction
with tab1:
    teams = sorted(elo.keys())
    c1, c2 = st.columns(2)
    with c1:
        team1 = st.selectbox("Team 1", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with c2:
        team2 = st.selectbox("Team 2", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

    if st.button("Predict", key="custom"):
        if team1 == team2:
            st.warning("Please select two different teams!")
        else:
            prob_dict = get_probs(team1, team2, elo, clf, team_forms)
            st.subheader(f"{team1} vs {team2}")
            for cls, prob in zip(clf.classes_, clf.predict_proba(
                pd.DataFrame([{
                    "elo_diff"  : elo.get(resolve(team1), 1500) - elo.get(resolve(team2), 1500),
                    "is_neutral": 1,
                    "home_form" : team_forms.get(resolve(team1), 0),
                    "away_form" : team_forms.get(resolve(team2), 0),
                    "form_diff" : team_forms.get(resolve(team1), 0) - team_forms.get(resolve(team2), 0),
                }]))[0]
            ):
                label = f"{team1} Win" if cls == "win" else (f"{team2} Win" if cls == "loss" else "Draw")
                st.write(f"**{label}:** {prob * 100:.1f}%")


# Tab 2: Group Stage Schedule
with tab2:
    st.subheader("2026 World Cup Group Stage Predictions")

    results_gs = []
    for match in FIXTURES:
        t1, t2    = match["team1"], match["team2"]
        prob_dict = get_probs(t1, t2, elo, clf, team_forms)
        predicted = max(prob_dict, key=prob_dict.get)
        prediction = (f"🟢 {t1}" if predicted == "win"
                      else f"🟢 {t2}" if predicted == "loss"
                      else "🟡 Draw")
        results_gs.append({
            "Date"      : match["date"],
            "Group"     : match["group"],
            "Team 1"    : t1,
            "Team 2"    : t2,
            "T1 Win %"  : f"{prob_dict.get('win',  0):.0%}",
            "Draw %"    : f"{prob_dict.get('draw', 0):.0%}",
            "T2 Win %"  : f"{prob_dict.get('loss', 0):.0%}",
            "Prediction": prediction,
        })

    schedule_df  = pd.DataFrame(results_gs)
    group_filter = st.selectbox("Filter by Group", ["All"] + sorted(schedule_df["Group"].unique()))
    filtered     = schedule_df if group_filter == "All" else schedule_df[schedule_df["Group"] == group_filter]
    st.dataframe(filtered, use_container_width=True)


# Tab 3: Tournament Simulator
with tab3:
    st.subheader("🏆 Full Tournament Simulation")
    st.caption("Model predicts every match from group stage through the Final.")

    with st.spinner("Simulating tournament..."):
        group_results    = simulate_group_stage(FIXTURES, elo, clf, team_forms)
        third_assign     = assign_third_place(group_results)
        knockout_results = simulate_knockout(group_results, third_assign, elo, clf, team_forms)

    # Group Standings
    st.markdown("### 📊 Predicted Group Standings")
    cols = st.columns(4)
    for i, g in enumerate(sorted(group_results.keys())):
        with cols[i % 4]:
            ranked = group_results[g]["ranked"]
            pts    = group_results[g]["pts"]
            st.markdown(f"**Group {g}**")
            for pos, team in enumerate(ranked):
                icon = "🥇" if pos == 0 else "🥈" if pos == 1 else "  "
                st.write(f"{icon} {team} — {pts[team]} pts")
            st.write("")

    # helper to display a round
    def show_round(title, match_nums):
        st.markdown(f"### {title}")
        for num in match_nums:
            r = knockout_results[num]
            st.write(
                f"**Match {num}:** {r['t1']} vs {r['t2']} → "
                f"🏆 **{r['winner']}** *(model confidence: {r['prob']}%)*"
            )

    show_round("⚔️ Round of 32",    list(range(73, 89)))
    show_round("🔟 Round of 16",    list(range(89, 97)))
    show_round("🏅 Quarter-Finals", list(range(97, 101)))
    show_round("🔥 Semi-Finals",    [101, 102])
    show_round("🥉 Third Place",    [103])

    # ── Final ────────────────────────────────
    final = knockout_results[104]
    st.markdown("---")
    st.markdown("## 🏆 THE FINAL")
    st.markdown(
        f"### {final['t1']}  🆚  {final['t2']}"
    )
    st.success(f"🎉 Predicted 2026 World Cup Champion: **{final['winner']}** ({final['prob']}% confidence)")