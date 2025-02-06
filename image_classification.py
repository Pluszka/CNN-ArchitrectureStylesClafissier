import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt



DATA_DIR = "data"
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

    data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=(256, 256), batch_size=32)

    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])

    data = data.map(lambda x, y: (x / 255.0, y))

    data_size = data.cardinality().numpy()
    train_size = int(data_size * 0.7)
    val_size = int(data_size * 0.2)
    test_size = data_size - train_size - val_size

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

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
    acc = BinaryAccuracy()
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(pre.result(), re.result(), acc.result())

    # img = cv2.imread('test.jpg')
    # plt.imshow(img)
    # plt.show()
    # resize = tf.image.resize(img, (256, 256))
    # plt.imshow(resize.numpy().astype(int))
    # plt.show()

    # yhat = model.predict(np.expand_dims(resize / 255, 0))


    model.save(os.path.join('models', 'imageclassifier.keras'))
    # new_model = load_model('models/imageclassifier.keras')
    # print(new_model.predict(np.expand_dims(resize / 255, 0)))

