import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('f1_dataset.csv')
data = data.dropna(subset=['QualiPosition', 'RacePosition'])

# Sort so each driver's races are in chronological order - essential for "recent form"
data = data.sort_values(['Driver', 'Year']).reset_index(drop=True)

# For each driver, calculate their average finishing position over their last 3 races
# .shift(1) makes sure we only look at PAST races, never the current one (avoiding "cheating")
data['RecentForm'] = (
    data.groupby('Driver')['RacePosition']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

# Drop rows where RecentForm couldn't be calculated (a driver's first ever race in the dataset)
data = data.dropna(subset=['RecentForm'])

# Now X has two inputs: qualifying position AND recent form
X = data[['QualiPosition', 'RecentForm']]
y = data['RacePosition']

model = LinearRegression()
model.fit(X, y)

print(f"Base finishing position: {model.intercept_:.2f}")
print(f"Effect of qualifying position: {model.coef_[0]:.3f}")
print(f"Effect of recent form: {model.coef_[1]:.3f}")

predictions = model.predict(X)
plt.scatter(y, predictions, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', label='Perfect prediction')
plt.xlabel('Actual Race Position')
plt.ylabel('Predicted Race Position')
plt.title('Predicting Race Result: Qualifying + Recent Form')
plt.legend()
plt.show()