import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import json

# 1. Generate Dummy Data for Predictive Maintenance (RUL)
np.random.seed(42)
n_samples = 300
# Features: temperature, vibration, pressure
X = np.random.rand(n_samples, 3) * 100
# Target: Remaining Useful Life (RUL) - Non-linear relationship with some noise
y = 500 - (X[:, 0] * 1.5) - (X[:, 1]**2 * 0.05) - (X[:, 2] * 0.8) + np.random.normal(0, 10, n_samples)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define and Train the Base Models
# Model A: AdaBoost Regressor
adaboost = AdaBoostRegressor(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# Model B: Stacking Regressor (SVR + RF -> Linear Regression)
estimators = [
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('svr', SVR(kernel='rbf', C=100, gamma='scale'))
]
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)
stacking.fit(X_train, y_train)

# 3. Generate Predictions on Validation Set
pred_ada = adaboost.predict(X_val)
pred_stack = stacking.predict(X_val)

# 4. Grid Search for Optimal Weights
weights_A = np.linspace(0, 1, 101)  # 0.0 to 1.0 with 0.01 step
mse_scores = []

for w_a in weights_A:
    w_b = 1.0 - w_a
    # Weighted average prediction
    pred_combined = (w_a * pred_ada) + (w_b * pred_stack)
    mse = mean_squared_error(y_val, pred_combined)
    mse_scores.append(mse)

# Find the best combination
best_idx = np.argmin(mse_scores)
best_w_a = weights_A[best_idx]
best_w_b = 1.0 - best_w_a
best_mse = mse_scores[best_idx]

# Save data to CSV for transparency
results_df = pd.DataFrame({
    'weight_adaboost': weights_A,
    'weight_stacking': 1 - weights_A,
    'mse': mse_scores
})
results_df.to_csv('ensemble_weights_optimization.csv', index=False)

print(f"Optimal AdaBoost Weight: {best_w_a:.2f}")
print(f"Optimal Stacking Weight: {best_w_b:.2f}")
print(f"Minimum MSE: {best_mse:.2f}")

# 5. Create Plotly Visualization
fig = go.Figure()

# Add the main curve
fig.add_trace(go.Scatter(
    x=weights_A, 
    y=mse_scores,
    mode='lines',
    name='Validation Error (MSE)',
    line=dict(width=3)
))

# Highlight the minimum point
fig.add_trace(go.Scatter(
    x=[best_w_a],
    y=[best_mse],
    mode='markers+text',
    name='Optimal Weight',
    marker=dict(size=12, color='red', symbol='star'),
    text=[f'Optimal (w={best_w_a:.2f})'],
    textposition="top right"
))

# Styling
fig.update_layout(
    title={"text": "Optimization of Ensemble Weights<br><span style='font-size: 14px; font-weight: normal;'>Finding lowest Validation Error between AdaBoost and Stacking Models</span>"},
    xaxis_title="Weight for AdaBoost (w_A)",
    yaxis_title="Mean Squared Error",
    showlegend=True,
    legend=dict(yanchor='top', y=0.95, xanchor='left', x=0.02),
    margin=dict(t=100)
)

# Save chart and metadata
chart_name = "ensemble_optimization.png"
fig.write_image(chart_name)

meta = {
    "caption": "Validation Error across different weight combinations for AdaBoost and Stacking Regressor",
    "description": "Line chart showing the convex curve of Mean Squared Error as the weight for AdaBoost increases from 0 to 1. The optimal point is highlighted with a red star."
}
with open(f"{chart_name}.meta.json", "w") as f:
    json.dump(meta, f)