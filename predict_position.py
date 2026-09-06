import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# Add the model's raw predicted position back onto the dataset
data['PredictedPosition'] = model.predict(X)

# For each race, rank drivers by their PREDICTED position (lowest predicted = best)
data['PredictedRank'] = data.groupby('Race')['PredictedPosition'].rank(method='first')

# Turn the predicted rank into the labels we actually care about
data['PredictedWinner'] = data['PredictedRank'] == 1
data['PredictedPodium'] = data['PredictedRank'] <= 3
data['PredictedTop10'] = data['PredictedRank'] <= 10

# Also calculate the ACTUAL labels, so we can check how often the model got it right
data['ActualWinner'] = data['RacePosition'] == 1
data['ActualPodium'] = data['RacePosition'] <= 3
data['ActualTop10'] = data['RacePosition'] <= 10

# How often did the model correctly predict the actual race winner?
winner_accuracy = (data['PredictedWinner'] & data['ActualWinner']).sum() / data['ActualWinner'].sum()
podium_accuracy = (data['PredictedPodium'] & data['ActualPodium']).sum() / data['ActualPodium'].sum()
top10_accuracy = (data['PredictedTop10'] & data['ActualTop10']).sum() / data['ActualTop10'].sum()

print(f"Correctly identified the actual winner: {winner_accuracy:.1%} of races")
print(f"Correctly identified an actual podium finisher: {podium_accuracy:.1%} of the time")
print(f"Correctly identified an actual top 10 finisher: {top10_accuracy:.1%} of the time")

# Save this enriched dataset so we can inspect specific races later
data.to_csv('f1_predictions.csv', index=False)
print("\nSaved predictions to f1_predictions.csv")