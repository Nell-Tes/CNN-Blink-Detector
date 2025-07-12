import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam

height, width = 26, 34

def readCsv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    imgs = np.empty((len(rows), height, width, 1), dtype=np.uint8)
    tgs = np.empty((len(rows), 1))

    for i, row in enumerate(rows):
        img = row['image']
        img = img.strip('[').strip(']').split(', ')
        im = np.array(img, dtype=np.uint8).reshape((height, width))
        im = np.expand_dims(im, axis=2)
        imgs[i] = im
        tgs[i] = 1 if row['state'] == 'open' else 0

    index = np.random.permutation(imgs.shape[0])
    return imgs[index], tgs[index]

def makeModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(height, width, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    xTrain, yTrain = readCsv('dataset.csv')
    xTrain = xTrain.astype('float32') / 255.0

    model = makeModel()

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    datagen.fit(xTrain)

    model.fit(datagen.flow(xTrain, yTrain, batch_size=32), steps_per_epoch=len(xTrain) // 32, epochs=50)
    model.save('blinkModel.hdf5')

if __name__ == "__main__":
    main()
