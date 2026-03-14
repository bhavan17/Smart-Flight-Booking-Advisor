# ================================================================
# EPOCH 3.0 — SMART BOOKING ADVISOR
# PES University AI & ML Datathon
# Role        : Regression Lead (P2)
# Model       : LightGBM Regressor
# Objective   : Predict flight prices and advise booking decisions
# Dataset     : 300,153 Indian domestic flight records
# ================================================================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# ----------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------

df_regression = pd.read_csv("df_regression.csv")
df_clean      = pd.read_csv("Clean_Dataset.csv")

print("=" * 50)
print("   EPOCH 3.0 — SMART BOOKING ADVISOR (P2)")
print("=" * 50)
print(f"  Total records  : {df_regression.shape[0]:,}")
print(f"  Total features : {df_regression.shape[1] - 1}")
print("=" * 50)

# ----------------------------------------------------------------
# 2. FEATURE AND TARGET SEPARATION
# ----------------------------------------------------------------

X = df_regression.drop(columns=['price'])
y = df_regression['price']

print("\nFeatures used for training:")
for col in X.columns:
    print(f"  - {col}")

# ----------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (80% train | 20% test)
# ----------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set : {X_train.shape[0]:,} records")
print(f"Testing set  : {X_test.shape[0]:,} records")

# ----------------------------------------------------------------
# 4. MODEL TRAINING — LightGBM Regressor
# ----------------------------------------------------------------

reg_model = LGBMRegressor(
    n_estimators  = 300,
    learning_rate = 0.05,
    max_depth     = 6,
    random_state  = 42,
    verbose       = -1
)

reg_model.fit(X_train, y_train)
print("\nModel training complete.")

# ----------------------------------------------------------------
# 5. MODEL EVALUATION
# ----------------------------------------------------------------

y_pred = reg_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("         REGRESSION MODEL METRICS")
print("=" * 50)
print(f"  RMSE  :  ₹{rmse:,.2f}")
print(f"  MAE   :  ₹{mae:,.2f}")
print(f"  R²    :  {r2:.4f}")
print("=" * 50)
print(f"  Model predicts price within ₹{mae:,.0f} on average.")
print(f"  Explains {r2*100:.2f}% of all price variation.")
print("=" * 50)

# ----------------------------------------------------------------
# 6. CROSS VALIDATION (5 Fold)
# ----------------------------------------------------------------

print("\nRunning 5-Fold Cross Validation... (takes ~1 min)")

cv_scores = cross_val_score(
    reg_model, X, y,
    scoring='r2',
    cv=5
)

print("\n" + "=" * 50)
print("       5-FOLD CROSS VALIDATION RESULTS")
print("=" * 50)
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}  :  R² = {score:.4f}")
print(f"\n  Mean R²  :  {cv_scores.mean():.4f}")
print(f"  Std Dev  :  ± {cv_scores.std():.4f}")
print("=" * 50)
print("  Low std dev = model is stable and consistent")
print("  High mean R² = model generalizes well to new data")
print("=" * 50)

# ----------------------------------------------------------------
# 7. SAVE TRAINED MODEL
# ----------------------------------------------------------------

with open("reg_model.pkl", "wb") as f:
    pickle.dump(reg_model, f)
print("\nTrained model saved as reg_model.pkl")

# ----------------------------------------------------------------
# 8. BUILD ENCODING MAPS
# ----------------------------------------------------------------

df_combined = df_clean.copy()
df_combined['airline_enc']          = df_regression['airline_enc']
df_combined['source_city_enc']      = df_regression['source_city_enc']
df_combined['destination_city_enc'] = df_regression['destination_city_enc']
df_combined['route_enc']            = df_regression['route_enc']
df_combined['class_enc']            = df_regression['class_enc']
df_combined['stops_enc']            = df_regression['stops_enc']
df_combined['departure_time_enc']   = df_regression['departure_time_enc']
df_combined['arrival_time_enc']     = df_regression['arrival_time_enc']
df_combined['duration_bucket_enc']  = df_regression['duration_bucket_enc']
df_combined['price_ratio']          = df_regression['price_ratio']
df_combined['route']                = df_combined['source_city'] + " → " + df_combined['destination_city']

airline_map                = df_combined.groupby('airline')['airline_enc'].first().to_dict()
source_map                 = df_combined.groupby('source_city')['source_city_enc'].first().to_dict()
destination_map            = df_combined.groupby('destination_city')['destination_city_enc'].first().to_dict()
class_map                  = df_combined.groupby('class')['class_enc'].first().to_dict()
stops_map                  = df_combined.groupby('stops')['stops_enc'].first().to_dict()
departure_map              = df_combined.groupby('departure_time')['departure_time_enc'].first().to_dict()
arrival_map                = df_combined.groupby('arrival_time')['arrival_time_enc'].first().to_dict()
route_enc_map              = df_combined.groupby('route')['route_enc'].first().to_dict()
duration_bucket_map        = df_combined.groupby('route')['duration_bucket_enc'].first().to_dict()
route_class_median_map     = df_combined.groupby(['route', 'class'])['price'].median().to_dict()
route_class_days_ratio_map = df_combined.groupby(['route', 'class', 'days_left'])['price_ratio'].mean().to_dict()
route_class_avg_ratio_map  = df_combined.groupby(['route', 'class'])['price_ratio'].mean().to_dict()

print("Encoding maps built successfully.")

# ----------------------------------------------------------------
# 9. SAVE ENCODING MAPS
# ----------------------------------------------------------------

handoff = {
    'airline_map'               : airline_map,
    'source_map'                : source_map,
    'destination_map'           : destination_map,
    'class_map'                 : class_map,
    'stops_map'                 : stops_map,
    'departure_map'             : departure_map,
    'arrival_map'               : arrival_map,
    'route_enc_map'             : route_enc_map,
    'duration_bucket_map'       : duration_bucket_map,
    'route_class_median_map'    : route_class_median_map,
    'route_class_days_ratio_map': route_class_days_ratio_map,
    'route_class_avg_ratio_map' : route_class_avg_ratio_map
}

with open("p2_handoff.pkl", "wb") as f:
    pickle.dump(handoff, f)
print("Encoding maps saved as p2_handoff.pkl")

# ----------------------------------------------------------------
# 10. SMART ADVISOR FUNCTION
# ----------------------------------------------------------------

def smart_advisor(source_city, destination_city, airline,
                  travel_class, days_left, stops,
                  departure_time, arrival_time, duration):

    route             = source_city + " → " + destination_city
    route_median      = route_class_median_map[(route, travel_class)]
    days_left_clamped = max(1, min(49, days_left))

    key = (route, travel_class, days_left_clamped)
    price_ratio = (
        route_class_days_ratio_map[key]
        if key in route_class_days_ratio_map
        else route_class_avg_ratio_map.get((route, travel_class), 1.0)
    )

    input_data = pd.DataFrame([{
        'airline_enc'          : airline_map[airline],
        'source_city_enc'      : source_map[source_city],
        'destination_city_enc' : destination_map[destination_city],
        'route_enc'            : route_enc_map[route],
        'class_enc'            : class_map[travel_class],
        'stops_enc'            : stops_map[stops],
        'departure_time_enc'   : departure_map[departure_time],
        'arrival_time_enc'     : arrival_map[arrival_time],
        'duration'             : duration,
        'duration_bucket_enc'  : duration_bucket_map[route],
        'days_left'            : days_left_clamped,
        'price_ratio'          : price_ratio
    }])

    predicted_price = reg_model.predict(input_data)[0]

    if predicted_price < route_median:
        advice     = "Book Now"
        confidence = round((1 - predicted_price / route_median) * 100, 2)
    else:
        advice     = "Wait"
        confidence = round((predicted_price / route_median - 1) * 100, 2)

    print("=" * 50)
    print(f"  Route           : {route}")
    print(f"  Airline         : {airline}")
    print(f"  Class           : {travel_class}")
    print(f"  Days to Depart  : {days_left}")
    print(f"  Predicted Price : ₹{predicted_price:,.0f}")
    print(f"  Route Median    : ₹{route_median:,.0f}")
    print(f"  Advice          : {advice}")
    print(f"  Confidence      : {confidence}%")
    print("=" * 50)

    return predicted_price, advice, confidence

# ----------------------------------------------------------------
# 11. LIVE DEMO
# ----------------------------------------------------------------

print("\n[DEMO 1] Economy — booked 45 days early")
smart_advisor("Delhi", "Mumbai", "SpiceJet", "Economy",
              45, "zero", "Early_Morning", "Morning", 2.17)

print("\n[DEMO 2] Business — last minute booking")
smart_advisor("Delhi", "Mumbai", "Vistara", "Business",
              2, "zero", "Morning", "Afternoon", 2.17)

print("\n[DEMO 3] Economy — booked 15 days early")
smart_advisor("Bangalore", "Chennai", "Indigo", "Economy",
              15, "zero", "Evening", "Night", 1.5)

# ----------------------------------------------------------------
# 12. SAMPLE PREDICTION TABLE
# ----------------------------------------------------------------

sample           = X_test.iloc[:10].copy()
actual_prices    = y_test.iloc[:10].values
predicted_prices = reg_model.predict(sample)
errors           = actual_prices - predicted_prices

print("\n" + "=" * 65)
print("         SAMPLE PREDICTIONS vs ACTUAL PRICES")
print("=" * 65)
print(f"  {'#':<4} {'Actual (₹)':>12} {'Predicted (₹)':>15} {'Error (₹)':>12} {'Accuracy':>10}")
print("-" * 65)
for i, (act, pred, err) in enumerate(zip(actual_prices, predicted_prices, errors)):
    accuracy = 100 - abs(err / act * 100)
    print(f"  {i+1:<4} {act:>12,.0f} {pred:>15,.0f} {err:>12,.0f} {accuracy:>9.1f}%")
print("-" * 65)
print(f"  {'Avg':<4} {actual_prices.mean():>12,.0f} {predicted_prices.mean():>15,.0f} {errors.mean():>12,.0f} {(100 - abs(errors/actual_prices*100)).mean():>9.1f}%")
print("=" * 65)

# ----------------------------------------------------------------
# 13. VISUALIZATION 1 — Feature Importance
# ----------------------------------------------------------------

importance = pd.Series(reg_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)

colors = [
    '#FF6B6B' if v == importance.max()
    else '#4ECDC4' if v >= importance.quantile(0.75)
    else '#45B7D1'
    for v in importance.values
]

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

bars = ax.barh(importance.index, importance.values,
               color=colors, edgecolor='none', height=0.6)

for bar, val in zip(bars, importance.values):
    ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f}', va='center', ha='left',
            color='white', fontsize=9, fontweight='bold')

ax.set_title('Feature Importance — What Drives Flight Prices?',
             fontsize=15, fontweight='bold', color='white', pad=20)
ax.set_xlabel('Importance Score', fontsize=11, color='#AAAAAA')
ax.tick_params(colors='white', labelsize=10)
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax.xaxis.grid(True, color='#2A2A2A', linewidth=0.8)
ax.set_axisbelow(True)

high = mpatches.Patch(color='#FF6B6B', label='Most Important')
mid  = mpatches.Patch(color='#4ECDC4', label='Important')
low  = mpatches.Patch(color='#45B7D1', label='Supporting')
ax.legend(handles=[high, mid, low], loc='lower right',
          facecolor='#1A1A2E', edgecolor='none',
          labelcolor='white', fontsize=9)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("feature_importance.png saved.")

# ----------------------------------------------------------------
# 14. VISUALIZATION 2 — Price vs Days Before Departure
# ----------------------------------------------------------------

df_clean['route'] = df_clean['source_city'] + " → " + df_clean['destination_city']
df_eco            = df_clean[df_clean['class'] == 'Economy']
top_routes        = df_eco['route'].value_counts().head(5).index.tolist()
df_filtered       = df_eco[df_eco['route'].isin(top_routes)]
df_grouped        = df_filtered.groupby(['route', 'days_left'])['price'].mean().reset_index()

colors_line = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

for i, route in enumerate(top_routes):
    data = df_grouped[df_grouped['route'] == route]
    ax.plot(data['days_left'], data['price'],
            label=route, linewidth=2.5,
            color=colors_line[i], alpha=0.9)
    ax.fill_between(data['days_left'], data['price'],
                    alpha=0.05, color=colors_line[i])

ax.axvline(x=20, color='yellow', linestyle='--',
           linewidth=1.5, alpha=0.7)
ax.text(21, ax.get_ylim()[1] * 0.95,
        '← Book Here!\n   Best Price Zone',
        color='yellow', fontsize=9, fontweight='bold')

ax.set_title('Flight Price vs Days Before Departure — Book Early, Pay Less!',
             fontsize=15, fontweight='bold', color='white', pad=15)
ax.set_xlabel('Days Left Before Departure', fontsize=11, color='#AAAAAA')
ax.set_ylabel('Average Price (₹)',          fontsize=11, color='#AAAAAA')
ax.tick_params(colors='white', labelsize=10)
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax.xaxis.grid(True, color='#2A2A2A', linewidth=0.8)
ax.yaxis.grid(True, color='#2A2A2A', linewidth=0.8)
ax.set_axisbelow(True)
ax.invert_xaxis()
ax.legend(loc='upper right', facecolor='#1A1A2E',
          edgecolor='none', labelcolor='white', fontsize=9)

plt.tight_layout()
plt.savefig("price_vs_days.png", dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("price_vs_days.png saved.")

# ----------------------------------------------------------------
# 15. VISUALIZATION 3 — Actual vs Predicted
# ----------------------------------------------------------------

sample_idx = np.random.choice(len(y_test), 500, replace=False)
actual     = np.array(y_test)[sample_idx]
predicted  = y_pred[sample_idx]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

ax.scatter(actual, predicted, alpha=0.4, color='#4ECDC4',
           edgecolors='none', s=25, label='Predictions')

min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
ax.plot([min_val, max_val], [min_val, max_val],
        color='#FF6B6B', linewidth=2,
        linestyle='--', label='Perfect Prediction Line')

ax.set_title('Actual vs Predicted Flight Prices',
             fontsize=15, fontweight='bold', color='white', pad=15)
ax.set_xlabel('Actual Price (₹)',    fontsize=11, color='#AAAAAA')
ax.set_ylabel('Predicted Price (₹)', fontsize=11, color='#AAAAAA')
ax.tick_params(colors='white', labelsize=10)
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax.xaxis.grid(True, color='#2A2A2A', linewidth=0.8)
ax.yaxis.grid(True, color='#2A2A2A', linewidth=0.8)
ax.set_axisbelow(True)

ax.text(0.05, 0.92, f'R²   = {r2:.4f}',
        transform=ax.transAxes, fontsize=12,
        color='#4ECDC4', fontweight='bold')
ax.text(0.05, 0.85, f'RMSE = ₹{rmse:,.0f}',
        transform=ax.transAxes, fontsize=12,
        color='#FFA07A', fontweight='bold')
ax.text(0.05, 0.78, f'MAE  = ₹{mae:,.0f}',
        transform=ax.transAxes, fontsize=12,
        color='#98D8C8', fontweight='bold')

ax.legend(facecolor='#1A1A2E', edgecolor='none',
          labelcolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("actual_vs_predicted.png saved.")

# ----------------------------------------------------------------
# 16. VISUALIZATION 4 — Price Distribution
# ----------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0F1117')

for ax in axes:
    ax.set_facecolor('#0F1117')
    ax.tick_params(colors='white', labelsize=10)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.xaxis.grid(True, color='#2A2A2A', linewidth=0.8)
    ax.yaxis.grid(True, color='#2A2A2A', linewidth=0.8)
    ax.set_axisbelow(True)

economy  = df_clean[df_clean['class'] == 'Economy']['price']
business = df_clean[df_clean['class'] == 'Business']['price']

axes[0].hist(economy,  bins=50, color='#4ECDC4',
             alpha=0.7, label='Economy',  edgecolor='none')
axes[0].hist(business, bins=50, color='#FF6B6B',
             alpha=0.7, label='Business', edgecolor='none')
axes[0].set_title('Price Distribution by Class',
                  fontsize=13, fontweight='bold', color='white', pad=12)
axes[0].set_xlabel('Price (₹)',           fontsize=10, color='#AAAAAA')
axes[0].set_ylabel('Number of Flights',   fontsize=10, color='#AAAAAA')
axes[0].legend(facecolor='#1A1A2E', edgecolor='none',
               labelcolor='white', fontsize=10)

airline_avg = df_clean.groupby('airline')['price'].mean().sort_values(ascending=False)
colors_bar  = ['#FF6B6B', '#FFA07A', '#4ECDC4', '#45B7D1', '#98D8C8', '#C3A6FF']

axes[1].bar(airline_avg.index, airline_avg.values,
            color=colors_bar, edgecolor='none')
axes[1].set_title('Average Price by Airline',
                  fontsize=13, fontweight='bold', color='white', pad=12)
axes[1].set_xlabel('Airline',           fontsize=10, color='#AAAAAA')
axes[1].set_ylabel('Average Price (₹)', fontsize=10, color='#AAAAAA')
axes[1].tick_params(axis='x', rotation=15)

for bar, val in zip(axes[1].patches, airline_avg.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 100,
                 f'₹{val:,.0f}',
                 ha='center', va='bottom',
                 color='white', fontsize=8, fontweight='bold')

plt.suptitle('Flight Price Analysis — Indian Domestic Routes',
             fontsize=15, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig("price_distribution.png", dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("price_distribution.png saved.")

# ----------------------------------------------------------------
# 17. VISUALIZATION 5 — Route wise Average Price
# ----------------------------------------------------------------

df_clean['route'] = df_clean['source_city'] + " → " + df_clean['destination_city']
route_avg = df_clean.groupby('route')['price'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

colors_route = [
    '#FF6B6B' if v == route_avg.max()
    else '#4ECDC4' if v >= route_avg.quantile(0.75)
    else '#45B7D1' if v >= route_avg.quantile(0.50)
    else '#98D8C8'
    for v in route_avg.values
]

bars = ax.bar(route_avg.index, route_avg.values,
              color=colors_route, edgecolor='none', width=0.6)

for bar, val in zip(bars, route_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f'₹{val:,.0f}',
            ha='center', va='bottom',
            color='white', fontsize=7, fontweight='bold')

ax.set_title('Average Flight Price by Route — Indian Domestic Flights',
             fontsize=15, fontweight='bold', color='white', pad=15)
ax.set_xlabel('Route',              fontsize=11, color='#AAAAAA')
ax.set_ylabel('Average Price (₹)', fontsize=11, color='#AAAAAA')
ax.tick_params(colors='white', labelsize=8)
ax.tick_params(axis='x', rotation=90)
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax.yaxis.grid(True, color='#2A2A2A', linewidth=0.8)
ax.set_axisbelow(True)

high = mpatches.Patch(color='#FF6B6B', label='Most Expensive')
mid1 = mpatches.Patch(color='#4ECDC4', label='Above Average')
mid2 = mpatches.Patch(color='#45B7D1', label='Average')
low  = mpatches.Patch(color='#98D8C8', label='Budget Friendly')
ax.legend(handles=[high, mid1, mid2, low],
          facecolor='#1A1A2E', edgecolor='none',
          labelcolor='white', fontsize=9)

plt.tight_layout()
plt.savefig("route_avg_price.png", dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("route_avg_price.png saved.")

# ----------------------------------------------------------------
# 18. VISUALIZATION 6 — Residual Plot
# ----------------------------------------------------------------

residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0F1117')

for ax in axes:
    ax.set_facecolor('#0F1117')
    ax.tick_params(colors='white', labelsize=10)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.xaxis.grid(True, color='#2A2A2A', linewidth=0.8)
    ax.yaxis.grid(True, color="#F7F7F7", linewidth=0.8)
    ax.set_axisbelow(True)

axes[0].scatter(y_pred, residuals, alpha=0.3,
                color='#4ECDC4', edgecolors='none', s=10)
axes[0].axhline(y=0, color='#FF6B6B', linewidth=2, linestyle='--')
axes[0].set_title('Residuals vs Predicted Price',
                  fontsize=13, fontweight='bold', color='white', pad=12)
axes[0].set_xlabel('Predicted Price (₹)', fontsize=10, color='#AAAAAA')
axes[0].set_ylabel('Residual Error (₹)',  fontsize=10, color='#AAAAAA')
axes[0].text(0.05, 0.92,
             'Random scatter = No systematic bias',
             transform=axes[0].transAxes,
             fontsize=9, color='#4ECDC4')

axes[1].hist(residuals, bins=80, color='#45B7D1',
             edgecolor='none', alpha=0.8)
axes[1].axvline(x=0, color='#FF6B6B', linewidth=2,
                linestyle='--', label='Zero Error')
axes[1].set_title('Distribution of Prediction Errors',
                  fontsize=13, fontweight='bold', color='white', pad=12)
axes[1].set_xlabel('Residual Error (₹)',    fontsize=10, color='#AAAAAA')
axes[1].set_ylabel('Number of Predictions', fontsize=10, color='#AAAAAA')
axes[1].text(0.05, 0.92,
             f'Mean Error : ₹{residuals.mean():,.0f}\nStd Dev    : ₹{residuals.std():,.0f}',
             transform=axes[1].transAxes,
             fontsize=9, color='#4ECDC4')
axes[1].legend(facecolor='#1A1A2E', edgecolor='none',
               labelcolor='white', fontsize=9)

plt.suptitle('Residual Analysis — Model Error Distribution',
             fontsize=15, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("residual_plot.png saved.")

# ----------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 50)
print("   EPOCH 3.0 — P2 COMPLETE SUMMARY")
print("=" * 50)
print(f"  Dataset       : 300,153 flight records")
print(f"  Model         : LightGBM Regressor")
print(f"  Features      : 12")
print(f"  RMSE          : ₹{rmse:,.2f}")
print(f"  MAE           : ₹{mae:,.2f}")
print(f"  R²            : {r2:.4f}")
print(f"  CV Mean R²    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 50)
print("  Saved Models:")
print("  → reg_model.pkl")
print("  → p2_handoff.pkl")
print("=" * 50)
print("  Charts Generated:")
print("  → feature_importance.png")
print("  → price_vs_days.png")
print("  → actual_vs_predicted.png")
print("  → price_distribution.png")
print("  → route_avg_price.png")
print("  → residual_plot.png")
print("=" * 50)
print("  Key Insights:")
print("  1. price_ratio is the #1 driver of flight price")
print("  2. Route and city pair are next biggest factors")
print("  3. Booking 20+ days early saves the most money")
print("  4. Last 15 days before departure = price spike")
print("  5. Business class costs 3-4x more than Economy")
print("  6. Vistara and Air India are most expensive airlines")
print("  7. Residuals are random = no systematic model bias")
print("=" * 50)
