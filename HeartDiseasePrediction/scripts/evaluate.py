import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_data, preprocess_data

def evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = joblib.load('models/rf_heart_model.pkl')
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == '__main__':
    evaluate()
