"""
real_time_recognition.py
Detecção (Haar), reconhecimento (LBPH) e landmarks (MediaPipe FaceMesh) ao vivo.
Permite ajustar parâmetros diretamente no topo.
"""

import cv2
import pickle
import numpy as np
import requests
import json
import os

# Try to import mediapipe, but make it optional
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe não disponível. Landmarks faciais não serão desenhados.")

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

# API Configuration
API_BASE_URL = 'https://assessor-virtual-api-684499909473.southamerica-east1.run.app/api'
API_LOGIN_EMAIL = 'admin@admin.com'
API_LOGIN_PASSWORD = 'admin'
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

# -----------------------
# API FUNCTIONS
# -----------------------
def authenticate_api():
    """
    Autentica na API e retorna o token de acesso.
    """
    try:
        login_url = f'{API_BASE_URL}/auth/login'
        payload = {
            'email': API_LOGIN_EMAIL,
            'senha': API_LOGIN_PASSWORD
        }
        response = requests.post(login_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            access_token = data.get('accessToken')
            print(f'✓ Autenticação bem-sucedida! Token obtido.')
            return access_token
        else:
            print(f'✗ Erro na autenticação: {response.status_code} - {response.text}')
            return None
    except requests.exceptions.RequestException as e:
        print(f'✗ Erro de conexão na autenticação: {e}')
        return None

def fetch_and_save_clients(access_token):
    """
    Busca todos os clientes da API e salva em clientes.json na raiz do projeto.
    """
    try:
        clientes_url = f'{API_BASE_URL}/clientes'
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.get(clientes_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            clientes_data = response.json()
            # Salva na raiz do projeto (um nível acima de Face_recognition_app)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(project_root, 'clientes.json')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(clientes_data, f, ensure_ascii=False, indent=2)
            
            print(f'✓ Dados de {len(clientes_data)} clientes salvos em: {output_path}')
            return True
        else:
            print(f'✗ Erro ao buscar clientes: {response.status_code} - {response.text}')
            return False
    except requests.exceptions.RequestException as e:
        print(f'✗ Erro de conexão ao buscar clientes: {e}')
        return False

# Variável para controlar se já autenticamos nesta sessão
api_authenticated = False
# -----------------------

# MediaPipe setup
if MEDIAPIPE_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=MP_MAX_NUM_FACES,
                                      min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                                      min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)
else:
    face_mesh = None

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
                                          minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE))    # 2) Aplica MediaPipe FaceMesh para landmarks
    if MEDIAPIPE_AVAILABLE and face_mesh:
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
                )# 3) Para cada face detectada pelo Haar, tenta reconhecer
    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200,200))

        label_id, confidence = recognizer.predict(face_resized)  # retorna (label, confidence)
        # Lembre: menor confidence -> melhor correspondência
        if confidence <= RECOGNITION_CONFIDENCE_THRESHOLD:
            name = id_to_name.get(label_id, 'Desconhecido')
            text = f'{name} ({confidence:.1f})'
            color = (0,200,0)
            
            # Autentica na API e busca dados dos clientes (apenas uma vez por sessão)
            if not api_authenticated:
                print(f'\n=== Reconhecimento válido detectado: {name} ===')
                print('Autenticando na API...')
                token = authenticate_api()
                if token:
                    print('Buscando dados dos clientes...')
                    if fetch_and_save_clients(token):
                        api_authenticated = True
                        print('=== Processo concluído com sucesso ===\n')
                    else:
                        print('=== Falha ao buscar dados dos clientes ===\n')
                else:
                    print('=== Falha na autenticação ===\n')
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
