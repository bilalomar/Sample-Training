import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Set the data directories
train_data_dir = 'path_to_train_data_directory'
test_data_dir = 'path_to_test_data_directory'

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
LEARNING_RATE = 0.001
EPOCHS = 10

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Data augmentation for test/validation images (only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load and prepare the train data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

# Load and prepare the test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

# Create EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)

# Save the model
model.save('binary_classification_efficientnet.h5')


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
loaded_model = load_model('binary_classification_efficientnet.h5')

# Load and preprocess an image for prediction
img_path = 'path_to_image_for_prediction.jpg'
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Perform prediction
predictions = loaded_model.predict(img_array)
class_labels = ['Class 0', 'Class 1']
predicted_class = np.argmax(predictions)
predicted_label = class_labels[predicted_class]

print(f'Predicted class: {predicted_class} - {predicted_label}')
