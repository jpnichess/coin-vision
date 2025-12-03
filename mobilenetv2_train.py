# # import tensorflow as tf
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# # from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# # TRAIN_DIR = "dataset/train"
# # VAL_DIR = "dataset/val"

# # # AUMENTAÇÃO AGRESSIVA (imprescindível)
# # train_datagen = ImageDataGenerator(
# #     preprocessing_function=preprocess_input,
# #     rotation_range=25,
# #     width_shift_range=0.15,
# #     height_shift_range=0.15,
# #     zoom_range=0.25,
# #     brightness_range=[0.4, 1.6],
# #     horizontal_flip=False
# # )

# # val_datagen = ImageDataGenerator(
# #     preprocessing_function=preprocess_input
# # )

# # train_gen = train_datagen.flow_from_directory(
# #     TRAIN_DIR,
# #     target_size=(224, 224),
# #     batch_size=32,
# #     class_mode='categorical',
# #     shuffle=True
# # )

# # val_gen = val_datagen.flow_from_directory(
# #     VAL_DIR,
# #     target_size=(224, 224),
# #     batch_size=32,
# #     class_mode='categorical',
# #     shuffle=False
# # )

# # # BASE DA REDE
# # base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
# # base.trainable = False   # Fase 1

# # # CABEÇA CUSTOMIZADA OTIMIZADA
# # x = GlobalAveragePooling2D()(base.output)
# # x = Dense(256, activation='relu')(x)
# # x = Dropout(0.3)(x)
# # output = Dense(3, activation='softmax')(x)

# # model = Model(inputs=base.input, outputs=output)

# # model.compile(
# #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
# #     loss='categorical_crossentropy',
# #     metrics=['accuracy']
# # )

# # # CALLBACKS INTELIGENTES
# # callbacks = [
# #     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
# #     ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
# #     ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
# # ]

# # # FASE 1 — TREINO DA CABEÇA
# # print("=== FASE 1: TREINO DA CABEÇA ===")
# # model.fit(
# #     train_gen,
# #     validation_data=val_gen,
# #     epochs=20,      # <- MÍNIMO
# #     callbacks=callbacks
# # )

# # # FASE 2 — FINE-TUNING DAS ÚLTIMAS CAMADAS
# # for layer in base.layers[-40:]:
# #     layer.trainable = True

# # model.compile(
# #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
# #     loss='categorical_crossentropy',
# #     metrics=['accuracy']
# # )

# # print("=== FASE 2: FINE-TUNING ===")
# # model.fit(
# #     train_gen,
# #     validation_data=val_gen,
# #     epochs=40,     # 40 epochs para estabilizar
# #     callbacks=callbacks
# # )

# # model.save("mobilenet_moedas.h5")
# # print("Modelo final salvo.")


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model

# TRAIN_DIR = "dataset/train"
# VAL_DIR = "dataset/val"

# # Data generator normal
# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)

# # 🔥 Aqui selecionamos SOMENTE a classe moeda_1_real
# classes_treino = ["moeda_1_real"]

# train_gen = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     classes=classes_treino,   # <<< SOMENTE 1-REAL
#     shuffle=True
# )

# val_gen = val_datagen.flow_from_directory(
#     VAL_DIR,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     classes=classes_treino,   # <<< SOMENTE 1-REAL
#     shuffle=False
# )

# # Agora o restante do seu modelo padrão
# base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base.trainable = False

# x = GlobalAveragePooling2D()(base.output)
# x = Dense(128, activation='relu')(x)
# output = Dense(1, activation='softmax')(x)   # Modelo só com uma classe

# model = Model(inputs=base.input, outputs=output)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_gen, validation_data=val_gen, epochs=10)

# model.save("modelo_1real.h5")



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# caminhos
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "mobilenet_moedas.h5"

# parâmetros simples
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
EPOCHS = 8
UNFREEZE_LAST_N = 20   # liberar as últimas 20 camadas do modelo (ajuste se quiser)

# geradores simples (rescale para 0-1)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# carrega modelo já treinado
model = load_model(MODEL_PATH)
print("Modelo carregado. Total de camadas:", len(model.layers))

# congela todas as camadas
for layer in model.layers:
    layer.trainable = False

# libera as últimas N camadas (simples e robusto)
for layer in model.layers[-UNFREEZE_LAST_N:]:
    layer.trainable = True

print(f"Últimas {UNFREEZE_LAST_N} camadas liberadas para treino.")

# recompila com learning rate baixo
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# treino rápido
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)

# salva sobre o mesmo arquivo (ou troque o nome se preferir)
model.save(MODEL_PATH)
print("Treino rápido concluído e modelo salvo.")
