import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/zomato_customer_churn.csv")
joblib.dump(model, "app/churn_model.pkl")


# Features and target
X = df[["OrderFrequency", "DaysSinceLastOrder", "AvgRating", "Complaints"]]
y = df["Churn"]

# Train model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "churn_model.pkl")

print("✅ churn_model.pkl saved successfully!")
