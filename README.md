# 🔐 Face Recognition App (Challenge IoT)

## 🫂Integrantes do grupo 
- Nome: Pedro Augusto Carneiro Barone Bomfim - RM: 99781
- Nome: João Pedro de Albuquerque Oliveira - RM: 551579
- Nome: Matheus Augusto Santos Rego - RM:551466
- Nome: Ian Cancian Nachtergaele - RM: 98387

## 🎯 Objetivo
Este projeto demonstra o uso de **Visão Computacional** para:
- Capturar imagens do rosto do usuário.
- Treinar um modelo de reconhecimento facial.
- Realizar a identificação em tempo real.

A ideia de contexto é um **assessor de investimentos** que usa o rosto como **senha biométrica**, garantindo maior segurança no acesso a informações sensíveis.

---


## 📹 Vídeo Demonstrativo
Você pode assistir ao vídeo explicativo do projeto clicando [aqui](https://link-do-seu-video.com).

---
## ⚙️ Dependências
Antes de rodar, instale o Python e as bibliotecas necessárias:

```bash
pip install opencv-python opencv-contrib-python numpy
```

## ▶️ Execução

Abra a pasta face_recognition_app no terminal do VS Code e siga os passos:

1. Capturar imagens

Captura fotos do rosto e salva em dataset/.

python capture_images.py --name "SeuNome" --max 60


Parâmetros:

--name: Nome da pessoa (usado como ID no treino).

--max: Número de imagens a capturar (padrão: 30).

2. Treinar o modelo

Gera o arquivo trainer.yml com os dados do reconhecimento facial.

python train_recognizer.py

3. Reconhecimento em tempo real

Abre a câmera e tenta identificar o rosto com base no modelo treinado.

python recognize_face.py

## ⚖️ Nota ética

O reconhecimento facial é uma tecnologia sensível, que pode trazer implicações de privacidade e segurança.
Este projeto é exclusivamente educacional e não deve ser usado em ambientes de produção sem:

Consentimento explícito dos usuários.

Armazenamento seguro de imagens e modelos.

Conformidade com legislações de proteção de dados (LGPD/GDPR).

## 📌 Observações

Todas as imagens ficam salvas localmente na pasta dataset/.

O modelo treinado fica no arquivo trainer.yml.

O projeto não envia dados para a internet, roda apenas na máquina local.


---

## 📄 .gitignore
```gitignore
# Arquivos Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Ambiente virtual (se usar venv)
venv/
env/

# Arquivos temporários
*.log

# Arquivos do Windows/macOS
.DS_Store
Thumbs.db

# Arquivo de treino gerado
trainer.yml

# Dataset (opcional - se não quiser subir as imagens reais)
dataset/
