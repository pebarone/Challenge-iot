"""
real_time_recognition.py
Detecção (Haar), reconhecimento (LBPH) e landmarks (MediaPipe FaceMesh) ao vivo.
Permite ajustar parâmetros diretamente no topo.
"""

import cv2
import pickle
import numpy as np
import mediapipe as mp

# -----------------------
# PARÂMETROS AJUSTÁVEIS
# -----------------------
# Haar Cascade (detecção)
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = 60

# Reconhecimento LBPH (limiar de confiança)
# OBS: para LBPH 'confidence' é uma DISTÂNCIA: menor -> melhor.
# Experimente:
#   <= 50 : estrito (poucas falsas ident.), pode rejeitar mesmo rostos corretos
#   ~ 60-90 : balanceado (recomendado)
#   > 100 : permissivo (maiores falsos positivos)
RECOGNITION_CONFIDENCE_THRESHOLD = 75.0

# MediaPipe (landmarks)
MP_MAX_NUM_FACES = 2
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5
# -----------------------

# Carrega cascade e modelo
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# carrega recognizer e labels
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('modelo_lbph.yml')
except Exception as e:
    raise RuntimeError('Erro ao carregar modelo LBPH. Execute train_recognizer.py antes.') from e

with open('labels.pkl', 'rb') as f:
    label_ids = pickle.load(f)
# invert mapping: id -> name
id_to_name = {v:k for k,v in label_ids.items()}

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=MP_MAX_NUM_FACES,
                                  min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                                  min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Não foi possível abrir a webcam.')

print('Iniciando reconhecimento - pressione "q" para sair.')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Detecção com Haar
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=HAAR_SCALE_FACTOR,
                                          minNeighbors=HAAR_MIN_NEIGHBORS,
                                          minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE))

    # 2) Aplica MediaPipe FaceMesh para landmarks
    mp_results = face_mesh.process(frame_rgb)

    # desenha landmarks detectados pelo MediaPipe (se houver)
    if mp_results.multi_face_landmarks:
        for face_landmarks in mp_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # 3) Para cada face detectada pelo Haar, tenta reconhecer
    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200,200))

        label_id, confidence = recognizer.predict(face_resized)  # retorna (label, confidence)
        # Lembre: menor confidence -> melhor correspondência
        if confidence <= RECOGNITION_CONFIDENCE_THRESHOLD:
            name = id_to_name.get(label_id, 'Desconhecido')
            text = f'{name} ({confidence:.1f})'
            color = (0,200,0)
        else:
            text = f'Desconhecido ({confidence:.1f})'
            color = (0,0,255)

        # retângulo e label
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # sobreposição com parâmetros atuais (útil para debugging/tuning)
    overlay_text = f'Haar(scale={HAAR_SCALE_FACTOR}, neigh={HAAR_MIN_NEIGHBORS}, minS={HAAR_MIN_SIZE}) | LBPH_thresh={RECOGNITION_CONFIDENCE_THRESHOLD} | MP_faces={MP_MAX_NUM_FACES}'
    cv2.putText(frame, overlay_text, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('Reconhecimento facial - q para sair', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
