import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
def load_data(file_path):
    columns = [
        "age", "job", "marital", "education", "default", "balance", "housing", "loan",
        "contact", "day", "month", "duration", "campaign", "pdays", "previous", 
        "poutcome", "y"
    ]
    data = pd.read_csv(file_path, sep=";", names=columns, skiprows=1)
    return data

# Preprocess data
def preprocess_data(data):
    # Encode categorical columns
    label_encoders = {}
    for column in ["job", "marital", "education", "default", "housing", "loan", 
                   "contact", "month", "poutcome", "y"]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Scale numerical columns
    scaler = StandardScaler()
    numerical_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data, label_encoders, scaler

# Train a model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    file_path = "bank.csv"  # Update this with your dataset file name
    data = load_data(file_path)
    data, label_encoders, scaler = preprocess_data(data)

    # Split data into features and target
    X = data.drop(columns=["y"])
    y = data["y"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
