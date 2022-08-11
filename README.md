# Résultats des Notebooks sous format HTML

Les résultats des notebooks en pages HTML :

<a href = https://deviluna29.github.io/oc_ingenieur-ia_p8/P08_00_notebook_analyse>Analyse du jeu de données</a>

<a href = https://deviluna29.github.io/oc_ingenieur-ia_p8/P08_01_notebook_modelisation>Modélisation et entraînement des différents modèles</a>

## Installation de l'environnement virtuel

Créer l'environnement à partir du fichier yaml
```bash
conda env create -f environment.yml
```

Activer l'environnement
```bash
conda activate projet_ml
```

## Téléchargement du jeu de données

Récupérer le jeu de données <a href = https://www.cityscapes-dataset.com/dataset-overview/>à cette adresse</a>

Dezipper les 2 fichiers dans les dossiers : 

- "data/gtFine/"
- "data/leftImg8bit/"

Chacun des dossiers doit contenir les 3 suivants :

- "test/"
- "train/"
- "val/"

## Application Web

Url de l'application : <a href = http://ocp8-segmentation.azurewebsites.net/>http://ocp8-segmentation.azurewebsites.net/</a>

Endpoint pour la segmentation d'une image :

http://ocp8-segmentation.azurewebsites.net/segment/
    
Placer une image dans le body de la requête. L'API retourne l'image segmentée.