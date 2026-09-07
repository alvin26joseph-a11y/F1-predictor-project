import pandas as pd

data = pd.read_csv('f1_predictions.csv')

# Pick a specific race AND year to inspect
race_name = 'Abu Dhabi Grand Prix'
year = 2023

race_data = data[(data['Race'] == race_name) & (data['Year'] == year)].sort_values('PredictedRank')

print(f"\n{year} {race_name} - Predicted vs Actual\n")
print(race_data[['Driver', 'QualiPosition', 'PredictedRank', 'RacePosition']].to_string(index=False))

predicted_winner = race_data[race_data['PredictedRank'] == 1]['Driver'].values[0]
actual_winner = race_data[race_data['RacePosition'] == 1]['Driver'].values[0]

print(f"\nModel predicted winner: {predicted_winner}")
print(f"Actual winner: {actual_winner}")
print(f"Correct: {predicted_winner == actual_winner}")