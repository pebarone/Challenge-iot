"""
capture_images.py
Captura imagens do rosto via webcam e salva em dataset/<nome_pessoa>/.
Permite ajustar parâmetros de detecção Haar Cascade.
"""

import cv2
import os
import argparse

# -----------------------
# PARÂMETROS AJUSTÁVEIS
# -----------------------
HAAR_SCALE_FACTOR = 1.1    # menor -> mais janelas, mais detecções; maior -> mais rápido, pode perder faces
HAAR_MIN_NEIGHBORS = 5     # maior -> menos falsos positivos, porém pode perder rostos pequenos
HAAR_MIN_SIZE = 60         # tamanho mínimo do rosto (px). Aumente para ignorar objetos pequenos
SAVE_FACE_SIZE = (200,200) # tamanho da imagem salva
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='Nome da pessoa (usado para a pasta).')
parser.add_argument('--max', type=int, default=60, help='Quantidade de fotos a capturar.')
args = parser.parse_args()

dataset_dir = 'dataset'
person_dir = os.path.join(dataset_dir, args.name)
os.makedirs(person_dir, exist_ok=True)

# Cascade do OpenCV (vem com a instalação)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Não foi possível abrir a webcam. Verifique conexão.')

count = len(os.listdir(person_dir))
print(f'Iniciando captura para "{args.name}". Já existem {count} imagens. Capturando até {args.max} novas.')

while count < args.max:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=HAAR_SCALE_FACTOR,
                                          minNeighbors=HAAR_MIN_NEIGHBORS,
                                          minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE))

    # desenha e salva maior face encontrada
    if len(faces) > 0:
        # seleciona a maior face (para caso de múltiplas)
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, SAVE_FACE_SIZE)
        filename = os.path.join(person_dir, f'{args.name}_{count:03d}.jpg')
        cv2.imwrite(filename, face_resized)
        count += 1

        # indica na tela
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, f'Salvo {count}/{args.max}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    else:
        cv2.putText(frame, 'Nenhuma face detectada', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow('Captura - pressione q para sair', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print('Interrompido pelo usuário.')
        break

print('Captura finalizada.')
cap.release()
cv2.destroyAllWindows()
