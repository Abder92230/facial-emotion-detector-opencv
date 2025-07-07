````markdown
# Détection faciale et reconnaissance des émotions avec OpenCV

> Un projet personnel de vision par ordinateur mêlant détection de visages et classification d’émotions à l’aide de modèles DNN et mini_XCEPTION.

## 👨‍💻 Réalisé par
**Abderrahmane Benali**

## 📸 Objectif du projet

Ce projet a été développé dans un cadre académique, mais l’ensemble du code, des optimisations et des scripts a été conçu et codé intégralement par moi-même. L’objectif était double :

- **Détecter automatiquement les visages** dans des images ou via une webcam, avec un modèle DNN optimisé.
- **Identifier les émotions faciales** en temps réel ou sur images fixes, à l’aide d’un modèle mini_XCEPTION pré-entraîné.

---

## 🧠 Modèles utilisés

- **Détection des visages** : `res10_300x300_ssd_iter_140000.caffemodel` + `deploy.prototxt` (modèle SSD basé sur Caffe)
- **Reconnaissance des émotions** : `fer2013_mini_XCEPTION.102-0.66.hdf5` (pré-entraîné sur le dataset FER-2013)

---

## ⚙️ Prérequis & Installation

### Système conseillé
- macOS (M1 ou Intel), Ubuntu ≥ 20.04

### Installation des dépendances

```bash
pip install -r requirements.txt
````

> Tous les packages sont compatibles avec ARM (testé sur MacBook Air M1 – 8Go RAM).

---

## 🚀 Comment exécuter

### 1. Détection d’émotions sur images

Place tes images dans `images/`, puis exécute :

```bash
python face_detectionV2.py
```

Les résultats sont enregistrés dans `output_images/`.

### 2. Détection d’émotions en temps réel

Lance :

```bash
python real_time_emotion_detection.py
```

* Appuie sur `c` pour capturer une image.
* Appuie sur `q` pour quitter.

---

## 📂 Structure du projet

```bash
.
├── face_detectionV2.py              # Détection sur images fixes
├── real_time_emotion_detection.py   # Détection en temps réel (caméra)
├── models/                          # Modèles pré-entraînés (.caffemodel, .hdf5)
├── images/                          # Images d'entrée
├── output_images/                   # Résultats annotés
├── requirements.txt                 # Dépendances Python
├── README.md                        # Présentation du projet
```

---

## 🧪 Résultats & Performances

* Précision élevée sur images nettes et centrées
* Détection correcte dans des conditions variées (angle, luminosité)
* Fonctionne en temps réel avec fluidité acceptable sur MacBook Air M1
* Reconnaissance fiable des émotions : 😐 😃 😢 😠 😱 😲 🤢

---

## 🔍 Limites connues

* Moins performant sur visages très petits ou très flous
* Quelques erreurs de classification en cas d'expressions ambiguës
* Performances en temps réel dépendantes de la machine

---

## 📈 Pistes d'amélioration

* Optimisation du traitement pour des machines plus modestes (via TensorFlow Lite)
* Interface utilisateur plus ergonomique
* Extension à d'autres jeux de données pour plus de robustesse

---

## 📚 Références

* [FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
* [OpenCV DNN module](https://docs.opencv.org/master/d6/d0f/group__dnn.html)
* [Mini\_XCEPTION architecture](https://arxiv.org/abs/1710.07557)

---

## 📝 Licence

Projet open-source sous licence MIT – libre à réutiliser ou adapter.
