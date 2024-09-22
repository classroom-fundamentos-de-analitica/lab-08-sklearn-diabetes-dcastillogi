import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train_data = pd.read_csv('train_dataset.csv')
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
test_data = pd.read_csv('test_dataset.csv')
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R²): {r2}")

# Save model
import pickle
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)