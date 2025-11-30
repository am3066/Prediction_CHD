#import pandas as pd
#import numpy as np
#import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, make_scorer, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- 1. Chargement des données et Nettoyage de 'famhist' ---
try:
    # Assurez-vous que le fichier CHDI.csv est dans le même dossier
    df = pd.read_csv("CHDI.csv", sep=';')
except FileNotFoundError:
    print("Erreur : Le fichier 'CHDI.csv' est introuvable. Veuillez vérifier le chemin.")
    exit()

# Harmonisation de la colonne catégorielle 'famhist'
if 'famhist' in df.columns:
    df['famhist'] = df['famhist'].astype(str).str.strip().str.lower().str.capitalize()
    # Remplacement des valeurs qui pourraient être 'nan' (manquantes) ou autres
    # Si des NaN sont présentes, elles seront gérées par le OneHotEncoder/ColumnTransformer
    # On peut les remplacer par 'Absent' si c'est la catégorie majoritaire ou les laisser
    # Le OneHotEncoder va les encoder s'il trouve 'Nan' comme catégorie.
    # Pour ce dataset spécifique, nous allons faire confiance à l'imputer/encoder plus tard.

# --- 2. Séparation des données ---
X = df.drop('chd', axis=1)
y = df['chd']

# Division stratifiée (important car 'chd' est déséquilibrée)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Définition des Pipelines de Prétraitement ---
numerical_features = ['sbp', 'ldl', 'adiposity', 'obesity', 'age']
categorical_features = ['famhist']

# Pipeline Numérique (Imputation + Standardisation)
numerical_pipeline = Pipeline([
    # Imputation par la médiane (plus robuste aux outliers)
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline Catégoriel (Encodage One-Hot)
categorical_pipeline = Pipeline([
    # Le OneHotEncoder va gérer les valeurs manquantes (NaN ou autres) en les traitant comme une catégorie
    # Nous ajoutons un imputer sur les catégories aussi, pour garantir que tout est géré.
    ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Absent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer pour combiner les prétraitements
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='drop'
)

# --- 4. Définition des Modèles et Hyperparamètres pour GridSearchCV ---
# Nous utilisons ImbPipeline pour inclure SMOTE avant le modèle
# SMOTE ne doit être appliqué que sur l'ensemble d'entraînement, ce que ImbPipeline assure.

# Le pipeline de base inclut Prétraitement, ACP et SMOTE
base_pipeline_steps = [
    ('preprocessor', preprocessor),
    # On inclut SMOTE pour traiter le déséquilibre de 'chd'
    ('smote', SMOTE(random_state=42)), 
    ('pca', PCA(random_state=42))
]

# Définition des modèles et de leurs grilles d'hyperparamètres
# 1. Régression Logistique (LR)
lr_pipeline = ImbPipeline(base_pipeline_steps + [('classifier', LogisticRegression(random_state=42, solver='liblinear'))])
lr_param_grid = {
    # PCA: nombre de composants à conserver
    'pca__n_components': [3, 4, 5, 0.95], # 0.95 conserve 95% de la variance
    # LR: paramètre de régularisation C
    'classifier__C': [0.1, 1.0, 10.0]
}

# 2. K-Nearest Neighbors (KNN)
knn_pipeline = ImbPipeline(base_pipeline_steps + [('classifier', KNeighborsClassifier())])
knn_param_grid = {
    # PCA: nombre de composants à conserver
    'pca__n_components': [3, 4, 5, 0.95],
    # KNN: nombre de voisins
    'classifier__n_neighbors': [5, 7, 9, 11]
}

# Liste des modèles à tester
grids = [
    (lr_pipeline, lr_param_grid, 'LogisticRegression'),
    (knn_pipeline, knn_param_grid, 'KNeighborsClassifier')
]

# Nous utiliserons le recall (rappel) pour la classe positive (chd=1) 
# comme métrique principale, car nous voulons minimiser les faux négatifs
# (ne pas détecter une maladie existante), ce qui est crucial dans le domaine médical.
scorer = make_scorer(recall_score, pos_label=1)

best_model = None
best_score = -1
best_name = ""

# --- 5. Optimisation des Hyperparamètres avec GridSearchCV ---
print("### Démarrage de l'Optimisation des Modèles avec GridSearchCV... ###")

for pipeline, param_grid, name in grids:
    print(f"\n-> Entraînement et optimisation pour {name}...")
    
    # GridSearchCV pour trouver les meilleurs hyperparamètres
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        scoring=scorer, # Utilisation du Recall pour la sélection
        cv=5, 
        verbose=1, 
        n_jobs=-1 # Utiliser tous les cœurs disponibles
    )
    
    # Entraînement sur les données
    grid_search.fit(X_train, y_train)
    
    # Évaluation sur l'ensemble de test
    y_pred = grid_search.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    print(f"\n--- Résultats {name} ---")
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Score de validation (Recall): {grid_search.best_score_:.4f}")
    print(f"Rapport de classification sur l'ensemble de test :\n{report}")
    
    # Sauvegarde du meilleur modèle
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_name = name

# --- 6. Sauvegarde du Meilleur Modèle ---
print(f"\n\n🏆 Meilleur modèle sélectionné : {best_name} avec un Recall de {best_score:.4f}")
model_filename = 'Model.pkl'

# Sauvegarde de l'intégralité du pipeline optimisé dans un fichier .pkl
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f"✅ Pipeline complet sauvegardé dans {model_filename}")
