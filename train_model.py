import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Dataset
df = pd.read_csv("HousePricePrediction.csv")

# Drop Id
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Convert categorical data
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df_encoded.drop("SalePrice", axis=1)
y = df_encoded["SalePrice"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# Save Model
joblib.dump(model, "house_model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

# Save metrics
metrics = {
    "r2": round(r2 * 100, 2),
    "mae": round(mae, 2),
    "rmse": round(rmse, 2)
}

joblib.dump(metrics, "metrics.pkl")

# Save graph data
graph_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

graph_df.to_csv("prediction_data.csv", index=False)

print("All files saved successfully")