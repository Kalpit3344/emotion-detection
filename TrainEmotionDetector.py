import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Data generators
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    'data/train', target_size=(48,48), batch_size=64,
    color_mode="grayscale", class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
    'data/test', target_size=(48,48), batch_size=64,
    color_mode="grayscale", class_mode='categorical')

# CNN Architecture
emotion_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2), Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2), Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'), Dropout(0.5),
    Dense(7, activation='softmax')
])

cv2.ocl.setUseOpenCL(False)
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy'])

# Train (takes ~2–4 hours on CPU, ~20 min on GPU)
emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64)

# Save model
model_json = emotion_model.to_json()
with open("model/emotion_model.json", "w") as f:
    f.write(model_json)
emotion_model.save_weights('model/emotion_model.h5')
print("Model saved!")