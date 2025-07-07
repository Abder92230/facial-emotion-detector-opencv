#!/bin/bash

# Mise à jour des paquets
echo "Mise à jour des paquets..."
sudo apt update && sudo apt upgrade -y

# Installer Python et pip
echo "Installation de Python3 et pip..."
sudo apt install -y python3 python3-pip

# Installer les dépendances Python
echo "Installation des dépendances Python..."
pip3 install -r requirements.txt

# Créer le dossier output_images s'il n'existe pas
echo "Création du dossier output_images..."
mkdir -p output_images

echo "Installation terminée ! Vous pouvez lancer les scripts avec run_image_detection.sh ou run_camera_detection.sh."
