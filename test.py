# Importation des bibliothèques nécessaires
import numpy as np
import pymongo
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Configuration initiale de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Movielens_NMF_Model")

# Connexion à MongoDB
client = pymongo.MongoClient('localhost:27017')
db = client['Movielens']
movies = db['movies']
users = db['users']

# Obtention des IDs uniques pour les films et les utilisateurs
movies_ids = movies.distinct('_id')  # Liste des IDs de films
users_ids = users.distinct('_id')  # Liste des IDs d'utilisateurs

# Création d'un DataFrame vide avec users_ids comme index et movies_ids comme colonnes, initialisé à 0
df = pd.DataFrame(0, index=users_ids, columns=movies_ids)

# Remplissage du DataFrame avec les évaluations
for u_id in users_ids:
    user_movies = users.find_one({'_id': u_id}, {'movies': 1})
    if user_movies and 'movies' in user_movies:
        for m in user_movies['movies']:
            m_id = m['movieid']
            rating = m['rating']
            if m_id in movies_ids:
                df.loc[u_id, m_id] = rating

# Conversion de df en DataFrame sparse pour une efficacité accrue
df_sparse = df.astype(pd.SparseDtype("float", 0))

# Démarrez une run MLflow
with mlflow.start_run():
    # Configuration des paramètres du modèle
    n_components = 20
    max_iter = 5000
    mlflow.log_param("n_components", n_components)
    mlflow.log_param("max_iter", max_iter)
    
    # Création et entraînement du modèle NMF avec df_sparse
    nmf = NMF(n_components=n_components, max_iter=max_iter)
    nmf.fit(df_sparse)
    
    # Log du modèle
    mlflow.sklearn.log_model(nmf, "nmf_model", registered_model_name="NMF_Model")
    
    # Evaluation du modèle
    U = nmf.transform(df_sparse)
    M = nmf.components_
    pred_matrix = np.dot(U, M)
    mse = mean_squared_error(df_sparse.values, pred_matrix)
    mae = mean_absolute_error(df_sparse.values, pred_matrix)

    # Log des métriques
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)

    # Affichage des résultats pour l'analyse
    print(f"MSE: {mse}, MAE: {mae}")


