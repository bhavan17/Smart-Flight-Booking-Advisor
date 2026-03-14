import pickle
import joblib
import pandas as pd
import os

# ── PATHS ──────────────────────────────────────────────────
base_dir    = os.path.dirname(os.path.abspath(__file__))
data_path   = base_dir
output_path = os.path.join(base_dir, "files")
model_path  = base_dir

encoders_file = os.path.join(model_path, "encoders.pkl")
reg_model_file = os.path.join(model_path, "reg_model.pkl")
clf_model_file = os.path.join(output_path, "clf_model_final.pkl")
clean_data_file = os.path.join(data_path, "Clean_Dataset.csv")

for required_file in [encoders_file, reg_model_file, clf_model_file, clean_data_file]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(f"Required file not found: {required_file}")

# ── LOAD EVERYTHING ────────────────────────────────────────
with open(encoders_file, 'rb') as f:
    encoders = pickle.load(f)

with open(reg_model_file, 'rb') as f:
    reg_model = pickle.load(f)

clf_model = joblib.load(clf_model_file)

# Keep classifier inference schema aligned with training in files/classifier_train.py
clf_feature_cols = [
    "airline_enc",
    "source_city_enc",
    "destination_city_enc",
    "route_enc",
    "class_enc",
    "stops_enc",
    "departure_time_enc",
    "arrival_time_enc",
    "duration",
    "duration_bucket_enc",
    "days_left",
]

df_raw        = pd.read_csv(clean_data_file,
                            usecols=['source_city','destination_city','price',
                                     'duration','days_left'])
df_raw['route'] = df_raw['source_city'] + " → " + df_raw['destination_city']
route_medians   = df_raw.groupby('route')['price'].median().to_dict()
global_median   = df_raw['price'].median()

# Auto-fill defaults from dataset averages
avg_duration  = round(df_raw['duration'].mean(), 2)
avg_days_left = int(df_raw['days_left'].mean())

# ── SMART ADVISOR ──────────────────────────────────────────
def smart_advisor(airline, source_city, destination_city,
                  travel_class, stops):

    route           = f"{source_city} → {destination_city}"
    duration        = avg_duration
    days_left       = avg_days_left
    departure_time  = "Morning"
    arrival_time    = "Afternoon"
    duration_bucket = "Medium"
    median          = route_medians.get(route, global_median)

    def encode(col, val):
        try:    return encoders[col].transform([val])[0]
        except: return 0

    reg_features = pd.DataFrame([{
        'airline_enc':          encode('airline',          airline),
        'source_city_enc':      encode('source_city',      source_city),
        'destination_city_enc': encode('destination_city', destination_city),
        'route_enc':            encode('route',            route),
        'class_enc':            encode('class',            travel_class),
        'stops_enc':            encode('stops',            stops),
        'departure_time_enc':   encode('departure_time',   departure_time),
        'arrival_time_enc':     encode('arrival_time',     arrival_time),
        'duration':             duration,
        'duration_bucket_enc':  encode('duration_bucket',  duration_bucket),
        'days_left':            days_left,
        'price_ratio':          0.9,
    }])

    clf_features = reg_features.drop(columns=["price_ratio"]).reindex(columns=clf_feature_cols, fill_value=0)

    predicted_price = reg_model.predict(reg_features)[0]
    decision_proba  = clf_model.predict_proba(clf_features)[0]
    confidence      = round(max(decision_proba) * 100, 1)
    is_good         = decision_proba[1] > 0.5
    decision        = "BOOK NOW 🟢" if is_good else "WAIT 🔴"
    saving          = median - predicted_price
    saving_text     = (f"You SAVE Rs.{saving:,.0f} vs usual!"
                       if saving > 0 else
                       f"Rs.{abs(saving):,.0f} MORE than usual.")

    print("\n" + "=" * 55)
    print("   ✈️   SMART BOOKING ADVISOR — EPOCH 3.0")
    print("=" * 55)
    print(f"  Route          : {route}")
    print(f"  Airline        : {airline}  |  Class : {travel_class}")
    print(f"  Stops          : {stops}")
    print("-" * 55)
    print(f"  Predicted Price : Rs. {predicted_price:,.0f}")
    print(f"  Usual Price     : Rs. {median:,.0f}")
    print(f"  Savings Info    : {saving_text}")
    print(f"  Our Advice      : {decision}")
    print(f"  Prediction Conf.: {confidence}%")
    print("=" * 55)

# ── USER INPUT — ONLY 5 QUESTIONS ─────────────────────────
def get_user_input():

    print("\n" + "=" * 55)
    print("   ✈️   WELCOME TO SMART BOOKING ADVISOR  ")
    print("=" * 55)

    # 1. Airline
    airlines = ["IndiGo", "Air India", "Vistara",
                "GO FIRST", "AirAsia", "SpiceJet"]
    print("\n1️⃣  Select Airline:")
    for i, a in enumerate(airlines, 1):
        print(f"   {i}. {a}")
    while True:
        try:
            airline = airlines[int(input("   Enter number: ")) - 1]
            break
        except:
            print("   ❌ Enter 1 to 6")

    # 2. Source City
    cities = ["Delhi", "Mumbai", "Bangalore",
              "Kolkata", "Hyderabad", "Chennai"]
    print("\n2️⃣  Select Source City:")
    for i, c in enumerate(cities, 1):
        print(f"   {i}. {c}")
    while True:
        try:
            source_city = cities[int(input("   Enter number: ")) - 1]
            break
        except:
            print("   ❌ Enter 1 to 6")

    # 3. Destination City
    remaining = [c for c in cities if c != source_city]
    print(f"\n3️⃣  Select Destination City:")
    for i, c in enumerate(remaining, 1):
        print(f"   {i}. {c}")
    while True:
        try:
            destination_city = remaining[int(input("   Enter number: ")) - 1]
            break
        except:
            print(f"   ❌ Enter 1 to {len(remaining)}")

    # 4. Class
    print("\n4️⃣  Select Travel Class:")
    print("   1. Economy")
    print("   2. Business")
    while True:
        try:
            travel_class = ["Economy", "Business"][int(input("   Enter number: ")) - 1]
            break
        except:
            print("   ❌ Enter 1 or 2")

    # 5. Stops
    print("\n5️⃣  Select Number of Stops:")
    print("   1. Zero (Direct Flight)")
    print("   2. One Stop")
    print("   3. Two or More")
    while True:
        try:
            stops = ["zero", "one", "two_or_more"][int(input("   Enter number: ")) - 1]
            break
        except:
            print("   ❌ Enter 1, 2 or 3")

    print("\n⏳ Analyzing your flight...")
    smart_advisor(airline, source_city, destination_city,
                  travel_class, stops)

    again = input("\n🔄 Try another flight? (y/n): ").strip().lower()
    if again == 'y':
        get_user_input()
    else:
        print("\n👋 Thank you!\n")

# ── RUN ────────────────────────────────────────────────────
get_user_input()
