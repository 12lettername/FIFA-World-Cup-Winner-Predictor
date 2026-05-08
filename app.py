import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


#  FUNCTIONS

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "win"
    elif row["home_score"] < row["away_score"]:
        return "loss"
    else:
        return "draw"

def get_form(team, date, df):
    past_matches = df[df["date"] < date]
    team_matches = past_matches[(past_matches["home_team"] == team) | (past_matches["away_team"] == team)]
    team_matches = team_matches.sort_values(by="date")
    last_5_matches = team_matches.tail(5)

    points = 0
    for _, match in last_5_matches.iterrows():
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



#  DATA LOADING & MODEL TRAINING as well as caching

@st.cache_data
def load_data():
    data = requests.get("https://raw.githubusercontent.com/martj42/international_results/master/results.csv").text
    df = pd.read_csv(StringIO(data), parse_dates=["date"])
    df = df[df["tournament"] != "Friendly"]
    df = df[df["date"] >= "2000-01-01"]
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["result"] = df.apply(get_result, axis=1)
    return df

@st.cache_data
def compute_elo(df, k=32, base=1500):
    elo = {}
    for _, row in df.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = elo.get(home, base)
        away_elo = elo.get(away, base)

        expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))

        if row["result"] == "win":
            actual = 1
        elif row["result"] == "draw":
            actual = 0.5
        else:
            actual = 0

        elo[home] = home_elo + k * (actual - expected)
        elo[away] = away_elo + k * ((1 - actual) - (1 - expected))
    return elo

@st.cache_data
def build_features(df, _elo_dict):
    records = []
    for _, row in df.iterrows():
        home_elo = _elo_dict.get(row["home_team"], 1500)
        away_elo = _elo_dict.get(row["away_team"], 1500)
        home_form = get_form(row["home_team"], row["date"], df)
        away_form = get_form(row["away_team"], row["date"], df)

        records.append({
            "elo_diff"  : home_elo - away_elo,
            "is_neutral": int(row["neutral"]),
            "home_form" : home_form,
            "away_form" : away_form,
            "form_diff" : home_form - away_form,
            "result"    : row["result"]
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



#  FIXTURES  

FIXTURES = [
    # Group A
    {"date": "Jun 11", "team1": "Mexico",                  "team2": "South Africa",           "group": "A"},
    {"date": "Jun 11", "team1": "South Korea",             "team2": "Czechia",                "group": "A"},
    {"date": "Jun 18", "team1": "Czechia",                 "team2": "South Africa",           "group": "A"},
    {"date": "Jun 18", "team1": "Mexico",                  "team2": "South Korea",            "group": "A"},
    {"date": "Jun 24", "team1": "Czechia",                 "team2": "Mexico",                 "group": "A"},
    {"date": "Jun 24", "team1": "South Africa",            "team2": "South Korea",            "group": "A"},
    # Group B
    {"date": "Jun 12", "team1": "Canada",                  "team2": "Bosnia and Herzegovina", "group": "B"},
    {"date": "Jun 13", "team1": "Qatar",                   "team2": "Switzerland",            "group": "B"},
    {"date": "Jun 18", "team1": "Switzerland",             "team2": "Bosnia and Herzegovina", "group": "B"},
    {"date": "Jun 18", "team1": "Canada",                  "team2": "Qatar",                  "group": "B"},
    {"date": "Jun 24", "team1": "Switzerland",             "team2": "Canada",                 "group": "B"},
    {"date": "Jun 24", "team1": "Bosnia and Herzegovina",  "team2": "Qatar",                  "group": "B"},
    # Group C
    {"date": "Jun 13", "team1": "Haiti",                   "team2": "Scotland",               "group": "C"},
    {"date": "Jun 13", "team1": "Brazil",                  "team2": "Morocco",                "group": "C"},
    {"date": "Jun 19", "team1": "Brazil",                  "team2": "Haiti",                  "group": "C"},
    {"date": "Jun 19", "team1": "Scotland",                "team2": "Morocco",                "group": "C"},
    {"date": "Jun 24", "team1": "Scotland",                "team2": "Brazil",                 "group": "C"},
    {"date": "Jun 24", "team1": "Morocco",                 "team2": "Haiti",                  "group": "C"},
    # Group D
    {"date": "Jun 12", "team1": "United States",           "team2": "Paraguay",               "group": "D"},
    {"date": "Jun 13", "team1": "Australia",               "team2": "Turkey",                 "group": "D"},
    {"date": "Jun 19", "team1": "Turkey",                  "team2": "Paraguay",               "group": "D"},
    {"date": "Jun 19", "team1": "United States",           "team2": "Australia",              "group": "D"},
    {"date": "Jun 25", "team1": "Turkey",                  "team2": "United States",          "group": "D"},
    {"date": "Jun 25", "team1": "Paraguay",                "team2": "Australia",              "group": "D"},
    # Group E
    {"date": "Jun 14", "team1": "Ivory Coast",             "team2": "Ecuador",                "group": "E"},
    {"date": "Jun 14", "team1": "Germany",                 "team2": "Curacao",                "group": "E"},
    {"date": "Jun 20", "team1": "Germany",                 "team2": "Ivory Coast",            "group": "E"},
    {"date": "Jun 20", "team1": "Ecuador",                 "team2": "Curacao",                "group": "E"},
    {"date": "Jun 25", "team1": "Curacao",                 "team2": "Ivory Coast",            "group": "E"},
    {"date": "Jun 25", "team1": "Ecuador",                 "team2": "Germany",                "group": "E"},
    # Group F
    {"date": "Jun 14", "team1": "Netherlands",             "team2": "Japan",                  "group": "F"},
    {"date": "Jun 14", "team1": "Sweden",                  "team2": "Tunisia",                "group": "F"},
    {"date": "Jun 20", "team1": "Netherlands",             "team2": "Sweden",                 "group": "F"},
    {"date": "Jun 20", "team1": "Tunisia",                 "team2": "Japan",                  "group": "F"},
    {"date": "Jun 25", "team1": "Japan",                   "team2": "Sweden",                 "group": "F"},
    {"date": "Jun 25", "team1": "Tunisia",                 "team2": "Netherlands",            "group": "F"},
    # Group G
    {"date": "Jun 15", "team1": "Iran",                    "team2": "New Zealand",            "group": "G"},
    {"date": "Jun 15", "team1": "Belgium",                 "team2": "Egypt",                  "group": "G"},
    {"date": "Jun 21", "team1": "Belgium",                 "team2": "Iran",                   "group": "G"},
    {"date": "Jun 21", "team1": "New Zealand",             "team2": "Egypt",                  "group": "G"},
    {"date": "Jun 26", "team1": "Egypt",                   "team2": "Iran",                   "group": "G"},
    {"date": "Jun 26", "team1": "New Zealand",             "team2": "Belgium",                "group": "G"},
    # Group H
    {"date": "Jun 15", "team1": "Saudi Arabia",            "team2": "Uruguay",                "group": "H"},
    {"date": "Jun 15", "team1": "Spain",                   "team2": "Cape Verde",             "group": "H"},
    {"date": "Jun 21", "team1": "Uruguay",                 "team2": "Cape Verde",             "group": "H"},
    {"date": "Jun 21", "team1": "Spain",                   "team2": "Saudi Arabia",           "group": "H"},
    {"date": "Jun 26", "team1": "Cape Verde",              "team2": "Saudi Arabia",           "group": "H"},
    {"date": "Jun 26", "team1": "Uruguay",                 "team2": "Spain",                  "group": "H"},
    # Group I
    {"date": "Jun 16", "team1": "France",                  "team2": "Senegal",                "group": "I"},
    {"date": "Jun 16", "team1": "Iraq",                    "team2": "Norway",                 "group": "I"},
    {"date": "Jun 22", "team1": "Norway",                  "team2": "Senegal",                "group": "I"},
    {"date": "Jun 22", "team1": "France",                  "team2": "Iraq",                   "group": "I"},
    {"date": "Jun 26", "team1": "Norway",                  "team2": "France",                 "group": "I"},
    {"date": "Jun 26", "team1": "Senegal",                 "team2": "Iraq",                   "group": "I"},
    # Group J
    {"date": "Jun 16", "team1": "Argentina",               "team2": "Algeria",                "group": "J"},
    {"date": "Jun 16", "team1": "Austria",                 "team2": "Jordan",                 "group": "J"},
    {"date": "Jun 22", "team1": "Argentina",               "team2": "Austria",                "group": "J"},
    {"date": "Jun 22", "team1": "Jordan",                  "team2": "Algeria",                "group": "J"},
    {"date": "Jun 27", "team1": "Algeria",                 "team2": "Austria",                "group": "J"},
    {"date": "Jun 27", "team1": "Jordan",                  "team2": "Argentina",              "group": "J"},
    # Group K
    {"date": "Jun 17", "team1": "Portugal",                "team2": "DR Congo",               "group": "K"},
    {"date": "Jun 17", "team1": "Uzbekistan",              "team2": "Colombia",               "group": "K"},
    {"date": "Jun 23", "team1": "Portugal",                "team2": "Uzbekistan",             "group": "K"},
    {"date": "Jun 23", "team1": "Colombia",                "team2": "DR Congo",               "group": "K"},
    {"date": "Jun 27", "team1": "Colombia",                "team2": "Portugal",               "group": "K"},
    {"date": "Jun 27", "team1": "DR Congo",                "team2": "Uzbekistan",             "group": "K"},
    # Group L
    {"date": "Jun 17", "team1": "Ghana",                   "team2": "Panama",                 "group": "L"},
    {"date": "Jun 17", "team1": "England",                 "team2": "Croatia",                "group": "L"},
    {"date": "Jun 23", "team1": "England",                 "team2": "Ghana",                  "group": "L"},
    {"date": "Jun 23", "team1": "Panama",                  "team2": "Croatia",                "group": "L"},
    {"date": "Jun 27", "team1": "Panama",                  "team2": "England",                "group": "L"},
    {"date": "Jun 27", "team1": "Croatia",                 "team2": "Ghana",                  "group": "L"},
]

# ── Name corrections ──────────────────────────
# If a team name in FIXTURES doesn't match the dataset,
# add it here: "fixture name" -> "dataset name"
# Run the debug check in tab2 to find mismatches!
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

#  LOAD & TRAIN

with st.spinner("Loading data and training model... (first run takes ~2 mins)"):
    df = load_data()
    elo = compute_elo(df)
    features_df = build_features(df, elo)
    clf = train_model(features_df)


#  UI

st.title("⚽ International Football Match Predictor")

tab1, tab2 = st.tabs(["🔮 Custom Prediction", "📅 WC 2026 Schedule"])


# ── Tab 1: Custom Prediction ─────────────────
with tab1:
    teams = sorted(list(elo.keys()))

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with col2:
        team2 = st.selectbox("Select Team 2", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

    if st.button("Predict", key="custom"):
        if team1 == team2:
            st.warning("Please select two different teams!")
        else:
            t1_elo  = elo.get(team1, 1500)
            t2_elo  = elo.get(team2, 1500)
            t1_form = get_form(team1, pd.Timestamp.today(), df)
            t2_form = get_form(team2, pd.Timestamp.today(), df)

            X = pd.DataFrame([{
                "elo_diff"  : t1_elo - t2_elo,
                "is_neutral": 1,
                "home_form" : t1_form,
                "away_form" : t2_form,
                "form_diff" : t1_form - t2_form
            }])

            probs     = clf.predict_proba(X)[0]
            classes   = clf.classes_
            prob_dict = dict(zip(classes, probs))

            st.subheader(f"Results: {team1} vs {team2}")
            for cls, prob in zip(classes, probs):
                if cls == "win":
                    label = f"{team1} Win"
                elif cls == "loss":
                    label = f"{team2} Win"
                else:
                    label = "Draw"
                st.write(f"**{label}:** {prob * 100:.1f}%")


# ── Tab 2: WC 2026 Schedule ──────────────────
with tab2:
    st.subheader("2026 World Cup Group Stage Predictions")

    # ── Uncomment to debug name mismatches ───
    # for m in FIXTURES:
    #     for team in [m["team1"], m["team2"]]:
    #         if resolve(team) not in elo:
    #             st.warning(f"NOT FOUND: {team} → {resolve(team)}")

    results = []
    for match in FIXTURES:
        t1     = match["team1"]
        t2     = match["team2"]
        t1_key = resolve(t1)
        t2_key = resolve(t2)

        t1_elo  = elo.get(t1_key, 1500)
        t2_elo  = elo.get(t2_key, 1500)
        t1_form = get_form(t1_key, pd.Timestamp("2026-06-01"), df)
        t2_form = get_form(t2_key, pd.Timestamp("2026-06-01"), df)

        X = pd.DataFrame([{
            "elo_diff"  : t1_elo - t2_elo,
            "is_neutral": 1,
            "home_form" : t1_form,
            "away_form" : t2_form,
            "form_diff" : t1_form - t2_form
        }])

        probs     = clf.predict_proba(X)[0]
        prob_dict = dict(zip(clf.classes_, probs))
        predicted = max(prob_dict, key=prob_dict.get)

        if predicted == "win":
            prediction = f"🟢 {t1}"
        elif predicted == "loss":
            prediction = f"🟢 {t2}"
        else:
            prediction = "🟡 Draw"

        results.append({
            "Date"      : match["date"],
            "Group"     : match["group"],
            "Team 1"    : t1,
            "Team 2"    : t2,
            "T1 Win %"  : f"{prob_dict.get('win',  0):.0%}",
            "Draw %"    : f"{prob_dict.get('draw', 0):.0%}",
            "T2 Win %"  : f"{prob_dict.get('loss', 0):.0%}",
            "Prediction": prediction
        })

    schedule_df = pd.DataFrame(results)

    group_filter = st.selectbox("Filter by Group", ["All"] + sorted(schedule_df["Group"].unique()))
    filtered = schedule_df if group_filter == "All" else schedule_df[schedule_df["Group"] == group_filter]
    st.dataframe(filtered, use_container_width=True)