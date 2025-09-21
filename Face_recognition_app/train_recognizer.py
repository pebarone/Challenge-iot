"""
train_recognizer.py
Varre a pasta dataset/ e treina um recognizer LBPH.
Salva: modelo (modelo_lbph.yml) e mapeamento labels (labels.pkl).
"""

import cv2
import os
import numpy as np
import pickle

# -----------------------
# PARÂMETROS AJUSTÁVEIS
# -----------------------
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = 60

# LBPH params: (radius, neighbors, grid_x, grid_y)
# radius: raio ao redor de cada pixel; neighbors: quantos pontos para comparação
# grid_x/grid_y: quantização espacial (maior -> mais detalhe)
LBPH_PARAMS = (1, 8, 8, 8)
# -----------------------

dataset_dir = 'dataset'
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

faces = []
labels = []
label_ids = {}
next_id = 0

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    label_ids[person_name] = next_id
    for filename in os.listdir(person_path):
        filepath = os.path.join(person_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # detecta face para garantir que estamos treinando só a região do rosto
        dets = face_cascade.detectMultiScale(img, scaleFactor=HAAR_SCALE_FACTOR,
                                             minNeighbors=HAAR_MIN_NEIGHBORS,
                                             minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE))
        if len(dets) > 0:
            # pega maior face
            dets = sorted(dets, key=lambda x: x[2]*x[3], reverse=True)
            x,y,w,h = dets[0]
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200,200))
            faces.append(face_resized)
            labels.append(next_id)
        else:
            # fallback: usa a imagem inteira (não ideal)
            face_resized = cv2.resize(img, (200,200))
            faces.append(face_resized)
            labels.append(next_id)
    next_id += 1

if len(faces) == 0:
    raise RuntimeError('Nenhuma face encontrada em dataset/. Execute capture_images.py primeiro.')

faces_np = np.array(faces)
labels_np = np.array(labels)

# Cria recognizer LBPH
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create(*LBPH_PARAMS)
except Exception as e:
    raise RuntimeError('cv2.face não disponível. Instale opencv-contrib-python.') from e

print('Treinando recognizer com', len(faces), 'amostras...')
recognizer.train(faces_np, labels_np)
recognizer.write('modelo_lbph.yml')

# salva mapeamento label -> nome
with open('labels.pkl', 'wb') as f:
    pickle.dump(label_ids, f)

print('Treinamento concluído. Modelo salvo em modelo_lbph.yml e labels.pkl')
print('Label IDs:', label_ids)
