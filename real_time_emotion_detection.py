import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import time
import os  # Pour manipuler les chemins de fichiers

# Charger le modèle DNN pour la détection de visages
face_net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt',
                                    'models/res10_300x300_ssd_iter_140000.caffemodel')

# Charger le modèle de reconnaissance d'expressions faciales (mini_XCEPTION)
emotion_model_path = 'models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

# Définir les labels pour les expressions faciales
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Chemin du dossier output_images
output_dir = '/Users/abderrahmanebenali/Downloads/OSTR/semestre9/mini_projet/output_images'
os.makedirs(output_dir, exist_ok=True)  # Crée le dossier si nécessaire

# Initialiser la caméra
cap = cv2.VideoCapture(0)  # 0 pour la caméra par défaut (intégrée au MacBook)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    ret, frame = cap.read()  # Lire une image depuis la caméra
    if not ret:
        print("Erreur lors de l'accès à la caméra.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    (h, w) = frame.shape[:2]

    # Préparation du blob pour la détection de visages
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Détection de visages
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:  # Seuil de confiance ajusté
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

            # Dessiner la boîte englobante et afficher l'étiquette de l'émotion
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label + f" ({emotion_probability*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Afficher le flux vidéo avec les prédictions
    cv2.imshow('Détection d\'émotions en temps réel', frame)

    # Appuyer sur 'c' pour capturer une image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captured_emotion_{timestamp}.jpg"
        file_path = os.path.join(output_dir, filename)
        cv2.imwrite(file_path, frame)
        print(f"Image capturée et enregistrée sous le nom : {file_path}")

    # Quitter la boucle avec la touche 'q'
    if key == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
