import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data, preprocess_data

def train():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler for later use
    joblib.dump(model, 'models/rf_heart_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Model training complete and saved.")

if __name__ == '__main__':
    train()
