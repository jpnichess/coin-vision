import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Caminhos das pastas

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# Data generators

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
TRAIN_DIR,
target_size=(224, 224),
batch_size=32,
class_mode='categorical',
shuffle=True,
seed=42
)

val_gen = val_datagen.flow_from_directory(
VAL_DIR,
target_size=(224, 224),
batch_size=32,
class_mode='categorical',
shuffle=False
)

# Carrega MobileNetV2 sem a cabeça final

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

# Cabeça customizada

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

# Treina o modelo

print("Treinando modelo...")
model.fit(
train_gen,
validation_data=val_gen,
epochs=7
)

# Salva o modelo

model.save("mobilenet_moedas.h5")
print("Modelo salvo como mobilenet_moedas.h5")
