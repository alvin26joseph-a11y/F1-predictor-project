import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('f1_dataset.csv')
data = data.dropna(subset=['QualiPosition', 'RacePosition'])
data = data.sort_values(['Driver', 'Year']).reset_index(drop=True)
data['RecentForm'] = (
    data.groupby('Driver')['RacePosition']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)
data_for_training = data.dropna(subset=['RecentForm'])

X = data_for_training[['QualiPosition', 'RecentForm']]
y = data_for_training['RacePosition']
model = LinearRegression()
model.fit(X, y)

latest_form = (
    data.sort_values(['Driver', 'Year'])
    .groupby('Driver')['RacePosition']
    .apply(lambda x: x.tail(3).mean())
)

# Today's actual 2026 Italian GP grid (from qualifying)
grid = [
    ('GAS', 1), ('RUS', 2), ('PIA', 3), ('LEC', 4), ('HAM', 5),
    ('VER', 6), ('ANT', 7), ('COL', 8), ('NOR', 9), ('LIN', 10),
    ('BOR', 11), ('BEA', 12), ('HUL', 13), ('LAW', 14), ('SAI', 15),
    ('OCO', 16), ('TSU', 17), ('ALB', 18), ('BOT', 19), ('PER', 20),
    ('ALO', 21), ('STR', 22)
]

predictions = []
missing_drivers = []

for driver, quali_pos in grid:
    if driver in latest_form.index:
        recent_form = latest_form[driver]
        input_df = pd.DataFrame([[quali_pos, recent_form]], columns=['QualiPosition', 'RecentForm'])
        predicted = model.predict(input_df)[0]
        predictions.append({'Driver': driver, 'QualiPosition': quali_pos, 'PredictedPosition': predicted})
    else:
        missing_drivers.append(driver)

predictions_df = pd.DataFrame(predictions).sort_values('PredictedPosition').reset_index(drop=True)
predictions_df['PredictedRank'] = predictions_df.index + 1

print("\n2026 Italian Grand Prix - Prediction\n")
print(predictions_df[['PredictedRank', 'Driver', 'QualiPosition']].to_string(index=False))

if missing_drivers:
    print(f"\nCould not predict (no history in dataset): {', '.join(missing_drivers)}")