# IPL Score Predictor - Save model.pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("IPL.csv")  # your dataset
print(df.head())

# ---------------------------
# 2. Feature Engineering
# ---------------------------
df['cumulative_runs'] = df.groupby(['mid', 'bat_team'])['runs'].cumsum()
df['cumulative_wickets'] = df.groupby(['mid', 'bat_team'])['wickets'].cumsum()

# overs are floats (e.g., 7.3 overs = 7.5 approx balls)
df['balls_bowled'] = (df['overs'].astype(int) * 6) + ((df['overs'] % 1) * 10).astype(int)
df['balls_left'] = 120 - df['balls_bowled']

df['wickets_left'] = 10 - df['cumulative_wickets']
df['current_run_rate'] = df['cumulative_runs'] / df['overs'].replace(0, 0.1)

# Final target
df['final_score'] = df['total']

# ---------------------------
# 3. Encode Categorical Features
# ---------------------------
categorical_cols = ['bat_team', 'bowl_team', 'venue']
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Combine encoded columns with main dataframe
df_model = pd.concat([df, encoded_df], axis=1)

# Drop unnecessary cols
drop_cols = ['mid', 'date', 'batsman', 'bowler', 'overs', 'total'] + categorical_cols
df_model = df_model.drop(columns=drop_cols)

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X = df_model.drop(columns=['final_score'])
y = df_model['final_score']

feature_cols = X.columns  # save for prediction later

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. Train Model
# ---------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# 6. Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Plot predicted vs actual
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("IPL Score Prediction: Actual vs Predicted")
plt.show()

# ---------------------------
# 7. Save Model + Encoder + Features
# ---------------------------
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(feature_cols, "features.pkl")

print("✅ Model, encoder, and feature columns saved!")
