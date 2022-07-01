# Résultats des Notebooks sous format HTML

Les résultats des notebooks en pages HTML :

...

## Configuration

...

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