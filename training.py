# ==== ALGORITMO TIRAR FOTOS (CONTINUAR DO 1001) ====

import cv2
import os

# Classes a capturar
classes = ["moeda_50_centavos", "moeda_1_real", "moeda_25_centavos"]

base_path = "dataset/val"
num_fotos = 200   # quantidade NOVA de fotos por classe

# Inicializa webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

for classe in classes:

    print(f"\n=== Capturando fotos para: {classe} ===")

    # Caminho da classe
    path_classe = os.path.join(base_path, classe)

    # Cria pasta se não existir
    os.makedirs(path_classe, exist_ok=True)

    # Conta quantas imagens já existem
    arquivos = [f for f in os.listdir(path_classe) if f.endswith(".jpg")]
    inicio = len(arquivos) + 1   # começa da próxima foto
    fim = inicio + num_fotos     # vai até +1000 fotos
    
    print(f"Iniciando da imagem {inicio} até {fim-1}")

    count = inicio

    while count < fim:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Pressione 'c' para capturar
        if key == ord('c'):
            img_name = os.path.join(path_classe, f"{classe}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Salvo: {img_name}")
            count += 1

        # Pressione 'q' para sair
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
