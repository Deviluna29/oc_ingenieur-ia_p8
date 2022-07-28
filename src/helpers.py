import numpy as np
import cv2
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

cats = {
        'void': [0, 1, 2, 3, 4, 5, 6],
        'flat': [7, 8, 9, 10],
        'construction': [11, 12, 13, 14, 15, 16],
        'object': [17, 18, 19, 20],
        'nature': [21, 22],
        'sky': [23],
        'human': [24, 25],
        'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
    }

cats_id = {
        'void': (0),
        'flat': (1),
        'construction': (2),
        'object': (3),
        'nature': (4),
        'sky': (5),
        'human':(6),
        'vehicle': (7)
    }

# Converti le masque pour le réduire à 8 catégories (one hote encoding)
def convert_mask(img):
        img = np.squeeze(img)
        mask = np.zeros((img.shape[0], img.shape[1], len(cats_id)), dtype='uint8')

        for i in range(-1, 34):
            for cat in cats:
                if i in cats[cat]:
                    mask[:,:,cats_id[cat]] = np.logical_or(mask[:,:,cats_id[cat]],(img==i))
                    break

        return np.array(mask, dtype='uint8')

# Fonction pour l'exemple d'augmentation des données
def augment_data(X, y):
        seq = iaa.Sequential([
            iaa.Sometimes( # Symétrie verticale sur 50% des images
                0.5,
                iaa.Fliplr(0.5)
            ),
            iaa.Sometimes( # Flou gaussien sur 50% des images
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.2))
            ),
            iaa.LinearContrast((0.8, 1.2)), # Modifie le contraste
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.1*255)), # Ajout d'un bruit gaussien
            iaa.Multiply((0.8, 1.2)), # Rend l'image plus sombre ou plus claire
            iaa.Affine( # Zoom, translation, rotation
                scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-15, 15)
            )
        ], random_order=True) # apply augmenters in random order

        new_X = []
        new_y = []

        for i in range(len(X)):
            img = X[i]
            mask = y[i]
            segmap = SegmentationMapsOnImage(mask, shape=img.shape)

            imag_aug_i, segmap_aug_i = seq(image=img, segmentation_maps=segmap)   
            new_X.append(imag_aug_i)
            new_y.append(segmap_aug_i.get_arr())

        new_X = np.array(new_X)
        new_y = np.array(new_y) 

        return new_X, new_y

# Prépare les données pour la segmentation avec model.predict()
def get_data_prepared(path_X_list, path_Y_list, dim):
  
    X = np.array([cv2.resize(cv2.imread(path_X), dim) for path_X in path_X_list])
    y = np.array([cv2.resize(convert_mask(cv2.imread(path_y, 0)), dim) for path_y in path_Y_list])

    if len(y.shape) == 3:
        y = np.expand_dims(y, axis = 3)
    X = X /255

    return X, y

# Retourne la synthèse d'un modèle sous la forme d'un dictionnaire
def add_model_in_synthese(loss_type, loss_score, mean_iou_score, training_time, predict_time):
    return {
            "loss type": loss_type,
            "loss score": loss_score,
            "mean_iou": mean_iou_score,
            "Training time": training_time,
            "Predict time": predict_time
        }

# Affiche graphiquement l'évolution de la fonction loss et de la métrique lors de l'entraînement d'un modèle
def draw_history(history):
    plt.subplots(1, 2, figsize=(15,4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['mean_iou'])
    plt.plot(history['val_mean_iou'])
    plt.title('model mean_iou')
    plt.ylabel('mean_iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()