import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Load dataset
df = pd.read_csv("Traffic.csv")

# 2. Copy dataset for modeling
model_df = df.copy()

# 3. Encode categorical columns
le_day = LabelEncoder()
le_target = LabelEncoder()

model_df["Day of the week"] = le_day.fit_transform(model_df["Day of the week"])
model_df["Traffic Situation"] = le_target.fit_transform(model_df["Traffic Situation"])

# 4. Define features and target
# We do NOT use Total here because the model without Total is more realistic.
X = model_df[["Date", "Day of the week", "CarCount", "BikeCount", "BusCount", "TruckCount"]]
y = model_df["Traffic Situation"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)

# 8. Save model and encoders
with open("traffic_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("label_encoder_day.pkl", "wb") as file:
    pickle.dump(le_day, file)

with open("label_encoder_target.pkl", "wb") as file:
    pickle.dump(le_target, file)

print("Model and encoders saved successfully.")