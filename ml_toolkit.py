import csv
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fonction pour lire les données depuis un fichier CSV
def read_data(file_path):
    # J'ouvre le fichier en mode lecture et je récupère les données ligne par ligne
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Je récupère la première ligne qui contient les noms des colonnes
        data = [row for row in reader]  # Je récupère toutes les autres lignes qui contiennent les données
    # Je convertis les données en tableau numpy pour faciliter les manipulations
    return header, np.array(data, dtype=float)

# Fonction pour nettoyer les données
def clean_data(data, columns_to_drop):
    # Je supprime les lignes qui contiennent des valeurs manquantes (NaN)
    data = data[~np.isnan(data).any(axis=1)]
    # Je supprime les colonnes que je ne souhaite pas utiliser
    data = np.delete(data, columns_to_drop, axis=1)
    return data

# Fonction pour normaliser les données
def normalize_data(X_train, X_test):
    # J'utilise StandardScaler pour normaliser les données afin d'améliorer la performance du modèle
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Fonction pour optimiser les hyperparamètres du modèle
def hyperparameter_tuning(X_train, y_train):
    # Je définis les valeurs des hyperparamètres que je souhaite tester
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # J'utilise GridSearchCV pour trouver la meilleure combinaison d'hyperparamètres
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# Fonction pour entraîner le modèle
def train_model(X_train, y_train, best_params):
    # Je crée le modèle avec les meilleurs hyperparamètres trouvés et je l'entraîne
    model = LogisticRegression(C=best_params['C'])
    model.fit(X_train, y_train)
    return model

# Fonction pour évaluer le modèle
def evaluate_model(model, X_test, y_test):
    # Je fais des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    # Je calcule la précision du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Je génère et affiche le rapport de classification et la matrice de confusion
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy


if __name__ == "__main__":
    # Je définis les paramètres initiaux
    file_path = "votre_fichier.csv"
    target_column_index = 3  # L'index de la colonne qui contient les valeurs cibles (à prédire)
    columns_to_drop = [0, 1]  # Les index des colonnes que je souhaite supprimer

    # Je lis les données depuis le fichier CSV
    header, data = read_data(file_path)
    # Je nettoie les données
    data = clean_data(data, columns_to_drop)
    # Je sépare les caractéristiques (X) et la cible (y)
    X = data[:, :-1]
    y = data[:, -1]
    # Je divise les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Je normalise les données
    X_train, X_test = normalize_data(X_train, X_test)
    # J'optimise les hyperparamètres du modèle
    best_params = hyperparameter_tuning(X_train, y_train)
    # J'entraîne le modèle
    model = train_model(X_train, y_train, best_params)
    # J'évalue le modèle
    evaluate_model(model, X_test, y_test)
