import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset we already built
data = pd.read_csv('f1_dataset.csv')

# Drop any rows with missing values (e.g. DNFs without a clean finishing position)
data = data.dropna(subset=['QualiPosition', 'RacePosition'])

X = data[['QualiPosition']]
y = data['RacePosition']

model = LinearRegression()
model.fit(X, y)

print(f"Base finishing position: {model.intercept_:.2f}")
print(f"Effect of each qualifying position: {model.coef_[0]:.3f}")

predictions = model.predict(X)
plt.scatter(X, y, alpha=0.3, label='Actual results')
plt.plot(X, predictions, color='red', label='Model prediction')
plt.xlabel('Qualifying Position')
plt.ylabel('Race Finishing Position')
plt.title('Predicting Race Result from Qualifying Position')
plt.legend()
plt.show()