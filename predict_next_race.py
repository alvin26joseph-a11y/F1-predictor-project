import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

fastf1.Cache.enable_cache('cache')

data = pd.read_csv('f1_dataset.csv')
data = data.sort_values(['Driver', 'Year']).reset_index(drop=True)

data['IsDNF'] = data['RacePosition'].isna().astype(int)

data['DNFRisk'] = (
    data.groupby('Driver')['IsDNF']
    .transform(lambda x: x.shift(1).expanding().mean())
)

data['RecentForm'] = (
    data.groupby('Driver')['RacePosition']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

finish_data = data.dropna(subset=['QualiPosition', 'RacePosition', 'RecentForm', 'DNFRisk'])
X_pos = finish_data[['QualiPosition', 'RecentForm', 'DNFRisk']]
y_pos = finish_data['RacePosition']
position_model = LinearRegression()
position_model.fit(X_pos, y_pos)

dnf_data = data.dropna(subset=['QualiPosition', 'DNFRisk'])
X_dnf = dnf_data[['QualiPosition', 'DNFRisk']]
y_dnf = dnf_data['IsDNF']
dnf_model = LogisticRegression()
dnf_model.fit(X_dnf, y_dnf)

latest_form = data.groupby('Driver')['RacePosition'].apply(lambda x: x.tail(3).mean())
latest_dnf_risk = data.groupby('Driver')['IsDNF'].apply(lambda x: x.mean())

# Pull the REAL qualifying grid automatically once it exists
year = 2026
round_number = 17  # CHECK this matches the Spanish GP round number for 2026 - run the schedule check first

quali = fastf1.get_session(year, round_number, 'Q')
quali.load(laps=False, telemetry=False, weather=False)
quali_results = quali.results

predictions = []
missing = []
for _, row in quali_results.iterrows():
    driver = row['Abbreviation']
    quali_pos = row['Position']
    if pd.isna(quali_pos):
        continue
    if driver in latest_form.index and driver in latest_dnf_risk.index:
        recent_form = latest_form[driver]
        dnf_risk = latest_dnf_risk[driver]

        pos_input = pd.DataFrame([[quali_pos, recent_form, dnf_risk]],
                                  columns=['QualiPosition', 'RecentForm', 'DNFRisk'])
        predicted_position = position_model.predict(pos_input)[0]

        dnf_input = pd.DataFrame([[quali_pos, dnf_risk]], columns=['QualiPosition', 'DNFRisk'])
        dnf_probability = dnf_model.predict_proba(dnf_input)[0][1]

        predictions.append({
            'Driver': driver, 'QualiPosition': quali_pos,
            'PredictedPosition': predicted_position, 'DNFRiskPercent': dnf_probability * 100
        })
    else:
        missing.append(driver)

predictions_df = pd.DataFrame(predictions).sort_values('PredictedPosition').reset_index(drop=True)
predictions_df['PredictedRank'] = predictions_df.index + 1

print(f"\n{year} Spanish Grand Prix - Prediction (using live qualifying results)\n")
print(predictions_df[['PredictedRank', 'Driver', 'QualiPosition', 'DNFRiskPercent']].round(1).to_string(index=False))
if missing:
    print("\nNo history:", missing)