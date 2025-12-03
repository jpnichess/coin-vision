# Biblioteca OpenCV de visão computacional
import cv2

# Biblioteca de manipulação numérica
import numpy as np

# Biblioteca utilizada para salvar um modelo Keras treinado e salvo anteriormente.
# from keras.models import load_model  

import tensorflow as tf

from tensorflow.keras.models import load_model
from collections import deque, Counter

from firebase_config import enviar_moeda


# Carrega o modelo treinado
modelo_keras = load_model("mobilenet_moedas.h5")

# Define a webcam como variável video
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Carregar modelo TFLite 
# Carrega o modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Pega detalhes dos tensores
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Carregar dados com o numpy devido a grande massa de caracteres
data = np.ndarray(shape = (1, 224, 224, 3), dtype = np.float32)

# Tipos de moeda carregados
classes = ["Moedas de 1 real", "Moedas 25 centavos", "Moeda 50 centavos"]

# Histórico de previsões para suavização temporal
historico = deque(maxlen=5)

def draw_corner_rect(img, x1, y1, x2, y2, color=(183, 195, 104), thickness=2, corner_len=20):
    # Canto superior esquerdo
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)

    # Canto superior direito
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)

    # Canto inferior esquerdo
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)

    # Canto inferior direito
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


# Função de pré processamento
def preProcess(img):
    #Aumenta os pixels da imagem
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    
    # Filtra imagem pelas bordas
    imgPre = cv2.Canny(imgPre, 90, 140)
    
    # Definir Kernel (pré processamento da IA)
    kernel = np.ones((4, 4), np.uint8)
    
    # Dilatar imagem
    imgPre = cv2.dilate(imgPre, kernel, iterations = 2)
    
    # Erosão morfológica (operação fundamental no processamento digital de imagens e é usada principalmente para reduzir os limites)
    impPre = cv2.erode(imgPre, kernel, iterations = 1)
    
    # Retorna valor final de pre processamento
    return imgPre

# Função de pré-processamento para o modelo Keras / MobileNetV2
def preProcessForModel(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = img / 255.0   # MobileNetV2 usa normalização 0–1
    img = np.expand_dims(img, axis=0)
    return img

# Função de reconhecimento
def DetectarMoeda(img):

    # Correção de iluminação (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab_corrigido = cv2.merge((l2, a, b))
    img = cv2.cvtColor(lab_corrigido, cv2.COLOR_LAB2BGR)

    # Pré-processamento correto para MobileNetV2
    imgMoeda = preProcessForModel(img)

    # Predição com Keras
    prediction = modelo_keras.predict(imgMoeda, verbose=0)
    
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]
    
    return classe, percent

# Enquanto tiver imagem é executado o bloco
while True:
    # Define img para ler o vídeo
    _,img = video.read()
    
    # Define o tamanho de px da img
    img = cv2.resize(img,(640, 480))
    
    # Chamar função de pré processamento da imagem
    imgPre = preProcess(img)
    
    # Criação da identificação de objeto e matriz de contornos
    countors, h1 = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    qtd = 0

    # Definir width, heigth, x e y e objets
    for cnt in countors:
        # Validação de area Evita que apenas alguns ruídos interfiram
        area = cv2.contourArea(cnt)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)

            # PADDING para melhorar o recorte das moedas
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            # Definir tamanho do retangulo, cor, etc
            draw_corner_rect(img, x1, y1, x2, y2)
            
            # Treinamento da Rede Neural para recorte
            recorte = img[y1:y2, x1:x2]
            classe, conf = DetectarMoeda(recorte)

            # THRESHOLD DE CONFIANÇA
            if conf < 0.70:
                classe = "Indefinido"

            # Adiciona classe ao histórico para suavização
            historico.append(classe)

            # SUAVIZAÇÃO TEMPORAL (classe mais votada nos últimos 5 frames)
            classe_suave = Counter(historico).most_common(1)[0][0]

            cv2.putText(img, str(classe_suave), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Contagem de valor 
            if classe_suave == "Moedas de 1 real": 
                qtd += 1
                enviar_moeda(100)
            if classe_suave == "Moedas 25 centavos": 
                qtd += 0.25
                enviar_moeda(25)
            if classe_suave == "Moeda 50 centavos": 
                qtd += 0.5
                enviar_moeda(50)
            if classe_suave == "Indefinido": 
                qtd = 0
                enviar_moeda(0)
            
    print(qtd)
    
    cv2.rectangle(img, (430, 30), (600, 80), (0, 0, 255), - 1)
    cv2.putText(img, f'R${qtd}', (440, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Mostra a imagem 
    cv2.imshow('IMG', img)
    
    # Mostra pré processamento da imagem 
    cv2.imshow('IMG PRE', imgPre)
    
    # Delay para demonstração da imagem
    cv2.waitKey(1)