from data_preprocessing import load_data, preprocess_data
from model_training import train_xgboost
from model_evaluation import evaluate_model
from config import DATA_PATH

def main():
    # Load and preprocess data
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train model
    model = train_xgboost(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
