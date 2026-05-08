# ⚽ 2026 World Cup Predictor

A machine learning web app that predicts match outcomes for the 2026 FIFA World Cup using historical match data, ELO ratings, and recent team form.

Built from scratch as a learning project — every function was written and understood before being added.

## Live Demo

## How It Works

### Data

- Loads 47,000+ international match results from martj42/international_results
- Filters to major tournaments only (World Cup, Euros, Copa América, AFCON, etc.)
- Excludes qualifiers from ELO computation to avoid confederation inflation

### ELO Rating System

Each team starts at 1500. After every match:

- The winner gains points, the loser loses points
- Upsets earn more points than expected wins
- Tournament importance is weighted (World Cup K=60, major tournaments K=50, etc.)

### Features Used

| Feature    | Description                                     |
| ---------- | ----------------------------------------------- |
| elo_diff   | Difference in ELO ratings between the two teams |
| is_neutral | Whether the match is at a neutral venue         |
| home_form  | Points earned in last 5 matches (team 1)        |
| away_form  | Points earned in last 5 matches (team 2)        |
| form_diff  | home_form minus away_form                       |

### Model

- Algorithm: Gradient Boosting Classifier (scikit-learn)
- Target: Win / Draw / Loss
- Train/Test split: 80% / 20%
- Accuracy: ~58-62% (vs 33% random baseline)

## Run Locally on your computer

### 1. Clone the repo

git clone https://github.com/12lettername/FIFA-World-Cup-Winner-Predictor
cd worldcup-predictor

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the app

streamlit run app.py

App opens at http://localhost:8501

First run takes 2-3 minutes to load data, compute ELO ratings, and train the model.
Subsequent runs are instant due to Streamlit caching.

## App Features

Tab 1 — Custom Prediction
Pick any two international teams and get Win / Draw / Loss probabilities instantly.

Tab 2 — Group Stage Schedule
All 72 group stage matches with predicted outcomes and probabilities, filterable by group.

Tab 3 — Tournament Simulator
Simulates the entire tournament from group stage through the Final and predicts the 2026 World Cup Champion.

## Possible Extensions

- Add a bracket visualizer for the knockout rounds
- Pull live FIFA rankings via API for more accurate ratings
- Add player injury data from Transfermarkt
- Simulate the tournament 1000 times and show win probabilities per team

## Data Source

martj42/international_results — a public dataset of international football results.
https://github.com/martj42/international_results

## Author

Built by Aryan Neogi(Fourteen Letter) as a data science project for the 2026 World Cup.
