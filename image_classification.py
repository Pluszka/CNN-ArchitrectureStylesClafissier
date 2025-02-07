import tensorflow as tf
import os
import cv2
import numpy as np
from keras.src.applications.vgg16 import VGG16
from keras.src.layers import GlobalAveragePooling2D
from keras.src.metrics import CategoricalAccuracy
from keras.src.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt

DATA_DIR = "data"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir(DATA_DIR))
EPOCHS = 20
IMAGE_EXTENSIONS = ["jpg", "jpeg", "bmp", "png"]

if __name__ == "__main__":
    for image_class in os.listdir(DATA_DIR):
        for image in os.listdir(os.path.join(DATA_DIR, image_class)):
            image_path = os.path.join(DATA_DIR, image_class, image)

            try:
                from PIL import Image

                try:
                    img = Image.open(image_path)
                    img.verify()
                    img = img.convert("RGB")
                    img.save(image_path)
                except (IOError, SyntaxError):
                    print(f"Removing invalid image: {image_path}")
                    os.remove(image_path)
                    continue

                file_extension = image.split('.')[-1].lower()

                if file_extension not in IMAGE_EXTENSIONS:
                    print(f"Image {image_path} has invalid extension {file_extension}")
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with image {image_path}: {e}")

    train_data = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.2, subset="training", seed=42
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.2, subset="validation", seed=42
    )

    # --- Normalizacja obrazów ---
    train_data = train_data.map(lambda x, y: (x / 255.0, y))
    val_data = val_data.map(lambda x, y: (x / 255.0, y))

    base_model = VGG16(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=[tensorboard_callback])

    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()
    for batch in val_data.as_numpy_iterator():
        X, y = batch
        y = to_categorical(y, num_classes=NUM_CLASSES)
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(pre.result(), re.result(), acc.result())

    test_img = cv2.imread('img.png')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, IMAGE_SIZE)
    plt.imshow(test_img)
    plt.show()

    test_img = np.expand_dims(test_img, axis=0).astype(np.float32) / 255.0
    yhat = model.predict(test_img)
    print(f"Prediction: {yhat}")


    model.save(os.path.join('models', 'architectureStylesClassifier.keras'))

