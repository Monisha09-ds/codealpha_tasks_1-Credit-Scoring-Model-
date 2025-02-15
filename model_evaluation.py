from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """Evaluates model and prints key metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print("🔹 Model Performance 🔹")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))
