import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import joblib
import os
import numpy as np

def load_data():
    # Load your dataset (replace with your actual data source)
    data = pd.read_csv(r'C:\quantum_trader\data\your_data.csv')  # Update with your actual data file path
    return data

def train_model(data):
    # Prepare features (X) and target (y)
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with your target variable
    y = data['target_column']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose a model
    model = RandomForestClassifier()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model accuracy: {accuracy:.2f}')
    
    # Save the model
    joblib.dump(model, 'trained_model.pkl')
    print("Model saved as 'trained_model.pkl'")

def initial_training():
    # Initialiser modellen
    model = SGDClassifier()

    # Anta at vi har en initial treningssett
    X_initial = np.array([[1, 2], [2, 3], [3, 4]])
    y_initial = np.array([0, 1, 0])

    # Tren modellen med initial data
    model.fit(X_initial, y_initial)

    # Lagre den initiale modellen
    joblib.dump(model, 'continuous_model.pkl')

def update_model_with_new_data():
    # Når nye data kommer inn
    new_data = np.array([[4, 5], [5, 6]])
    new_labels = np.array([1, 0])

    # Last inn den eksisterende modellen
    model = joblib.load('continuous_model.pkl')

    # Oppdater modellen med nye data
    model.partial_fit(new_data, new_labels)

    # Lagre den oppdaterte modellen
    joblib.dump(model, 'continuous_model.pkl')

if __name__ == "__main__":
    # Change directory to where the script is located
    os.chdir(r"C:\quantum_trader\backend")
    
    data = load_data()
    train_model(data)
    initial_training()
    update_model_with_new_data()