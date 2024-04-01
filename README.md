# MovieWise_Recommender_System
## MovieWise Recommender System
Ce référentiel contient les fichiers et le code pour un système de recommandation de films, développé dans le cadre d'un projet d'entraînement. Le système utilise divers modèles de machine learning pour recommander des films aux utilisateurs en fonction de leurs préférences.

## Contenu
mlruns/: Ce répertoire contient les résultats des différentes expériences de modélisation.
your_mflow_uri: Emplacement du stockage des artefacts de modèle.
Movie_propre.ipynb: Notebook Jupyter contenant le code pour le prétraitement des données sur les films.
test.py: Script Python pour tester les modèles.
traitement_db.ipynb: Notebook Jupyter pour le traitement de la base de données.
ml-1m/: Répertoire contenant les données sur les utilisateurs et les films.

## Développement
Pour développer localement, vous pouvez cloner ce référentiel en utilisant la commande suivante :
bash
Copy code
git clone https://github.com/aminataguia/MovieWise_Recommender_System.git

## Environnement
Ce projet nécessite Python 3.x et les bibliothèques suivantes :
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Utilisation
Vous pouvez utiliser ce système de recommandation en exécutant les scripts fournis dans le dossier mlruns/ pour entraîner les modèles et en utilisant ensuite le script test.py pour tester les performances des modèles.
