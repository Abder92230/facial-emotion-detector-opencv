import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Charger le modèle DNN pour la détection de visages
face_net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt',
                                    'models/res10_300x300_ssd_iter_140000.caffemodel')

# Charger le modèle de reconnaissance d'expressions faciales (mini_XCEPTION)
emotion_model_path = 'models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

# Définir les labels pour les expressions faciales
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Créer un dossier pour stocker les résultats si ce n'est pas déjà fait
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Charger et traiter chaque image dans le dossier "images"
for image_name in os.listdir("images"):
    img_path = os.path.join("images", image_name)
    img = cv2.imread(img_path)
    
    if img is None:
        continue  # Ignorer si le fichier n'est pas une image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h, w) = img.shape[:2]

    # Préparation du blob pour la détection de visages
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Détection de visages
    face_net.setInput(blob)
    detections = face_net.forward()

    # Boucle sur chaque détection de visage
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:  # Seuil de confiance ajusté pour détecter plus de visages
            # Extraire les coordonnées de la boîte englobante du visage
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # S'assurer que les coordonnées sont dans les limites de l'image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Extraire le visage et le prétraiter pour le modèle
            face = gray[y1:y2, x1:x2]
            if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                continue  # Ignorer les visages trop petits

            face = cv2.resize(face, (64, 64))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Prédiction de l'expression faciale
            preds = emotion_classifier.predict(face)[0]
            emotion_probability = np.max(preds)
            emotion_label_arg = np.argmax(preds)
            label = emotions[emotion_label_arg]

            # Dessiner la boîte englobante et afficher l'étiquette de l'émotion sur l'image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Couleur verte
            cv2.putText(img, label + f" ({emotion_probability*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)  # Couleur verte

    # Enregistrer le résultat dans le dossier "output_images"
    output_path = os.path.join(output_folder, "processed_" + image_name)
    cv2.imwrite(output_path, img)
    print(f"Image processed and saved as {output_path}")

# Fermeture des fenêtres (si jamais on réutilise cv2.imshow à l'avenir)
cv2.destroyAllWindows()
