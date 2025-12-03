# Biblioteca OpenCV de visão computacional
import cv2

# Biblioteca de manipulação numérica
import numpy as np

# Biblioteca utilizada para salvar um modelo Keras treinado e salvo anteriormente.
# from keras.models import load_model   # <-- Fica comentado pois não usaremos .h5

import tensorflow as tf

# Define a webcam como variável video
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ======== NOVO: Carregar modelo TFLite ========
# Carrega o modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Pega detalhes dos tensores
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# ==============================================

# Carregar dados com o numpy devido a grande massa de caracteres
data = np.ndarray(shape = (1, 224, 224, 3), dtype = np.float32)

# Tipos de moeda carregados
classes = ["Moedas de 1 real", "Moedas 25 centavos", "Moeda 50 centavos"]

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
    
# Função de reconhecimento
def DetectarMoeda(img):
    # Tamanho da foto 224 px e com isso pega essa semelhança para moedas
    imgMoeda = cv2.resize(img, (224, 224))
    # Carrega as arrays do numpy
    imgMoeda = np.asarray(imgMoeda)
    #Processo de normalização do teachblemachine
    imgMoedaNormilize = (imgMoeda.astype(np.float32) / 127.0) - 1
    # Dados da primeira posição é igual a imagem interpretada
    data[0] = imgMoedaNormilize

    # ======== NOVO: Executa predição com TFLite ========
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    # ====================================================

    # Índice para identificação das classes  
    index = np.argmax(prediction)
    # Pegar a resposta do Keras
    percent = prediction[0][index]
    # Passar o índice da classe
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

            # Definir tamanho do retangulo, cor, etc
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Treinamento da Rede Neural para recorte
            recorte = img[y:y + h,x:x + w]
            classe, conf = DetectarMoeda(recorte)
            cv2.putText(img, str(classe), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Contagem de valor 
            if classe == "Moedas de 1 real": qtd += 1
            
            if classe == "Moedas 25 centavos": qtd += 0.25
            
            if classe == "Moeda 50 centavos": qtd += 0.5
            
    print(qtd)
    
    cv2.rectangle(img, (430, 30), (600, 80), (0, 0, 255), - 1)
    cv2.putText(img, f'R${qtd}', (440, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Mostra a imagem 
    cv2.imshow('IMG', img)
    
    # Mostra pré processamento da imagem 
    cv2.imshow('IMG PRE', imgPre)
    
    # Delay para demonstração da imagem
    cv2.waitKey(1)