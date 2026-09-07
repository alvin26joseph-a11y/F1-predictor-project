import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression

fastf1.Cache.enable_cache('cache')

# STEP 1: Rebuild the trained model using historical data (same as before)
data = pd.read_csv('f1_dataset.csv')
data = data.dropna(subset=['QualiPosition', 'RacePosition'])
data = data.sort_values(['Driver', 'Year']).reset_index(drop=True)
data['RecentForm'] = (
    data.groupby('Driver')['RacePosition']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)
data = data.dropna(subset=['RecentForm'])

X = data[['QualiPosition', 'RecentForm']]
y = data['RacePosition']
model = LinearRegression()
model.fit(X, y)

# STEP 2: Get each driver's CURRENT recent form (their last 3 known results)
latest_form = (
    data.sort_values(['Driver', 'Year'])
    .groupby('Driver')['RacePosition']
    .apply(lambda x: x.tail(3).mean())
)

# STEP 3: Load the qualifying results for the race we want to predict
# Change year/round_number to the race you want to predict
year = 2023
round_number = 22  # Abu Dhabi 2023, as an example - swap for a real upcoming race

quali = fastf1.get_session(year, round_number, 'Q')
quali.load(laps=False, telemetry=False, weather=False)
quali_results = quali.results

# STEP 4: Build a prediction table using ONLY pre-race information
predictions = []
for _, row in quali_results.iterrows():
    driver = row['Abbreviation']
    quali_pos = row['Position']
    if driver in latest_form.index and pd.notna(quali_pos):
        recent_form = latest_form[driver]
        predicted = model.predict([[quali_pos, recent_form]])[0]
        predictions.append({'Driver': driver, 'QualiPosition': quali_pos, 'PredictedPosition': predicted})

predictions_df = pd.DataFrame(predictions).sort_values('PredictedPosition').reset_index(drop=True)
predictions_df['PredictedRank'] = predictions_df.index + 1

print(f"\nPredictions for {year} Round {round_number} (before the race)\n")
print(predictions_df[['PredictedRank', 'Driver', 'QualiPosition']].to_string(index=False))