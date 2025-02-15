import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV
from config import XGB_PARAMS

def train_xgboost(X_train, y_train):
    """Trains XGBoost model with hyperparameter tuning and saves it."""
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

    # Hyperparameter tuning
    grid_search = GridSearchCV(xgb_model, XGB_PARAMS, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Save model
    joblib.dump(best_model, "credit_model.pkl")
    print("✅ Model saved as 'credit_model.pkl'")

    return best_model
