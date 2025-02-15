import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load and Preprocess Data
def load_and_preprocess_data(file_path):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(file_path)

    # Drop ID column (not useful for prediction)
    df.drop(columns=['ID'], inplace=True)

    # Convert percentage columns to numerical values
    percentage_cols = ['TLBalHCPct', 'TLSatPct', 'TLOpenPct', 'TLOpen24Pct']
    for col in percentage_cols:
        df[col] = df[col].str.replace('%', '', regex=True).astype(float)

    # Convert object columns to numeric
    object_cols = ['TLSum', 'TLMaxSum']
    for col in object_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values (fill with median)
    df.fillna(df.median(), inplace=True)

    return df

# Step 2: Train Model
def train_model(df):
    """Trains an XGBoost model."""
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Train model with early stopping
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    return model

# Step 3: Save Model
def save_model(model, filename="credit_model.pkl"):
    """Saves the trained model."""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Step 4: Run Training Pipeline
def main():
    file_path = "E:\\Data Science Projects\\CodeAlpha_Assignments\\Credit_Scoring_Model\\a_Dataset_CreditScoring.xlsx - Sheet1.csv"  # Update with actual file path
    df = load_and_preprocess_data(file_path)
    model = train_model(df)
    save_model(model)

# Run the script
if __name__ == "__main__":
    main()
