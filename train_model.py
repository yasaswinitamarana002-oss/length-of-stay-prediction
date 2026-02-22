import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("patients data.csv")

# Convert dates
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df['departure_date'] = pd.to_datetime(df['departure_date'])

# Create Length of Stay
df['Length_of_Stay'] = (df['departure_date'] - df['arrival_date']).dt.days

# Remove unnecessary columns
df = df.drop(columns=['patient_id', 'name', 'arrival_date', 'departure_date'])

X = df.drop("Length_of_Stay", axis=1)
y = df["Length_of_Stay"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['service'])
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "los_prediction_model.pkl")

print("Model trained and saved successfully!")