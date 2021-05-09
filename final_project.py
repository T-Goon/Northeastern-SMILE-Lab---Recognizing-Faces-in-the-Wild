from operator import mod
import numpy as np
import pandas as pd
from os import listdir, path
from PIL import Image, ImageOps
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as lr
from sklearn.utils.testing import ignore_warnings
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# creates positive samples from the data set
def save_positive_samples(df, i, type, model):
    trainData = []
    counter = 0
    c=0

    for index, row in df.iterrows():

        try:
            # get all images in the 2 directories
            imgs_p1 = listdir(row['p1'])
            imgs_p2 = listdir(row['p2'])

            # all combinations of the images in either directory
            for file1 in imgs_p1:

                img1 = Image.open(row['p1'] + '/' + file1)
                for file2 in imgs_p2:
                    img2 = Image.open(row['p2'] + '/' + file2)

                    # print(len(get_embedding(model,np.asarray(img1.resize((160, 160), Image.ANTIALIAS)))) )

                    trainData.append(
                        np.concatenate((get_embedding(model,np.asarray(img1.resize((160, 160), Image.ANTIALIAS))),
                        get_embedding(model,np.asarray(img2.resize((160, 160), Image.ANTIALIAS)))
                    )))
                    counter +=1
                    c+=1

                    if(counter >= 1000):
                        print('pos {}: {}'.format(type, c))
                        counter = 0

                    # trainData.append(
                    #     np.concatenate(
                    #         (np.asarray(ImageOps.grayscale(img1.resize((128, 128), Image.ANTIALIAS))),
                    #          np.asarray(ImageOps.grayscale(img2.resize((128, 128), Image.ANTIALIAS))))))

        except FileNotFoundError:
            pass

    # save all the images in a numpy file
    data = np.array(trainData)
    np.save("{}_images_pos{}.npy".format(type, i), data)


# creates negative samples from the data set
def save_negative_samples(df, i, type, model):
    trainData = []
    person1 = []
    person2 = []

    counter = 0
    c=0

    for index, row in df.iterrows():

        try:
            # get all the images in the 2 directories
            imgs_p1 = listdir(row['p1'])
            imgs_p2 = listdir(row['p2'])

            # all combinations of the images in either directory
            for file1 in imgs_p1:

                img1 = Image.open(row['p1'] + '/' + file1)
                for file2 in imgs_p2:
                    img2 = Image.open(row['p2'] + '/' + file2)

                    person1.append(get_embedding(model,np.asarray(img1.resize((160, 160), Image.ANTIALIAS))))
                    person2.append(get_embedding(model,np.asarray(img2.resize((160, 160), Image.ANTIALIAS))))

                    # resize image and turn into an np array
                    # person1.append(np.asarray(ImageOps.grayscale(img1.resize((128, 128), Image.ANTIALIAS))))
                    # person2.append(np.asarray(ImageOps.grayscale(img2.resize((128, 128), Image.ANTIALIAS))))

                    counter +=1
                    c+=1

                    if(counter >= 1000):
                        print('neg {}: {}'.format(type, c))
                        counter = 0

        except FileNotFoundError:
            pass

    np.random.shuffle(person2)

    trainData = np.concatenate(
        (person1, person2),
        axis=1
    )
    # save all the images in a numpy file
    data = np.array(trainData)
    np.save("{}_images_neg{}.npy".format(type, i), data)


# Turns positive training examples into a large numpy array
# array is saved in training_images.npy
# Format [person 1 or person 2, 
#         sample #,
#         rows of image,
#         columns of image,
#         rbg]
#
# Data comes from the train.zip file
# Extract contents of train.zip to a data/train
# Gets path from data/train_relationships.csv
def convert_to_numpy():
    # read csv for training samples and print dataframe
    df = pd.read_csv('data/train_relationships.csv')
    model = keras.models.load_model('facenet_keras.h5')
    # print(df)

    # iterate over everything in the csv
    # for index, row in df.iterrows():
    #     print(row['p1'], row['p2'])

    # set up correct path to images
    df['p1'] = 'data/train/' + df['p1']
    df['p2'] = 'data/train/' + df['p2']

    # random sample of 5% of the data set
    p = .05
    for i in range(3):
        b = df.sample(frac=p)
        save_positive_samples(b, i, "training", model)

    print('training pos done')

    for i in range(3):
        b = df.sample(frac=p)
        save_negative_samples(b, i, 'training', model)

    print('training neg done')

    for i in range(1):
        b = df.sample(frac=p)
        save_positive_samples(b, i, "testing", model)

    print('testing pos done')

    for i in range(1):
        b = df.sample(frac=p)
        save_negative_samples(b, i, 'testing', model)

    print('testing neg done')


def shallow():
    if (not path.exists('training_images_pos0.npy') and
            not path.exists('training_image_neg0.npy')):
        convert_to_numpy()

        # (# samples, 256, 128, 3)
    train_pos = np.load('training_images_pos0.npy')
    train_neg = np.load('training_images_neg0.npy')
    test_pos = np.load('testing_images_pos0.npy')
    test_neg = np.load('testing_images_neg0.npy')
    print(train_pos.shape)
    print(train_neg.shape)
    print(test_pos.shape)
    print(test_neg.shape)

    train_pos = preprocessing.minmax_scale(train_pos)
    train_neg = preprocessing.minmax_scale(train_neg)
    test_pos = preprocessing.minmax_scale(test_pos)
    test_neg = preprocessing.minmax_scale(test_neg)

    pos_label = np.ones(train_pos.shape[0])
    neg_label = np.zeros(train_neg.shape[0])

    trainX = np.concatenate((train_pos, train_neg))
    trainY = np.concatenate((pos_label, neg_label))

    # Flip images & add flipped version
    trainXFlip = np.concatenate((trainX[:, 128:], trainX[:, :128]), axis=1)
    trainX = np.concatenate((trainX, trainXFlip))
    trainY = np.concatenate((trainY, trainY))

    idxs = np.random.permutation(trainX.shape[0])

    trainX = trainX[idxs]
    trainY = trainY[idxs]
    
    clf = lr().fit(trainX, trainY)

    pos_label_test = np.ones(test_pos.shape[0])
    neg_label_test = np.zeros(test_neg.shape[0])

    testX = np.concatenate((test_pos, test_neg))
    testY = np.concatenate((pos_label_test, neg_label_test))

    idxs = np.random.permutation(testX.shape[0])

    testX = testX[idxs]
    testY = testY[idxs]

    print('Accuracy: ', clf.score(testX, testY))

    print('ROC: ', sklearn.metrics.roc_auc_score(testY, clf.predict_proba(testX)[:, 1]))

    # plt.imshow(train_pos[0])
    # plt.show()

def deep():
    if (not path.exists('training_images_pos0.npy') and
            not path.exists('training_image_neg0.npy')):
        convert_to_numpy()

        # (# samples, 256, 128, 3)
    train_pos = np.load('training_images_pos0.npy')
    train_neg = np.load('training_images_neg0.npy')
    test_pos = np.load('testing_images_pos0.npy')
    test_neg = np.load('testing_images_neg0.npy')
    print(train_pos.shape)
    print(train_neg.shape)
    print(test_pos.shape)
    print(test_neg.shape)

    # flatten images
    train_pos = preprocessing.minmax_scale(train_pos)
    train_neg = preprocessing.minmax_scale(train_neg)
    test_pos = preprocessing.minmax_scale(test_pos)
    test_neg = preprocessing.minmax_scale(test_neg)

    pos_label = np.ones(train_pos.shape[0])
    neg_label = np.zeros(train_neg.shape[0])

    trainX = np.concatenate((train_pos, train_neg))
    trainY = np.concatenate((pos_label, neg_label))

    # Flip images & add flipped version
    trainXFlip = np.concatenate((trainX[:, 128:], trainX[:, :128]), axis=1)
    trainX = np.concatenate((trainX, trainXFlip))
    trainY = np.concatenate((trainY, trainY))

    idxs = np.random.permutation(trainX.shape[0])

    trainX = trainX[idxs]
    trainY = trainY[idxs]
    trainY = tf.one_hot(trainY, 2)

    pos_label_test = np.ones(test_pos.shape[0])
    neg_label_test = np.zeros(test_neg.shape[0])

    testX = np.concatenate((test_pos, test_neg))
    testY = np.concatenate((pos_label_test, neg_label_test))

    testXFlip = np.concatenate((testX[:, 128:], testX[:, :128]), axis=1)
    testX = np.concatenate((testX, testXFlip))
    testY = np.concatenate((testY, testY))

    idxs = np.random.permutation(testX.shape[0])

    testX = testX[idxs]
    testY = testY[idxs]
    testY = tf.one_hot(testY, 2)

    model = Sequential()
    # model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
    #                 activation='relu',
    #                 input_shape=()))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    model.add(Dense(100, activation='relu', input_shape=testX.shape))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy', 'AUC'])

    # train model
    model.fit(trainX, trainY,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(testX, testY))

    score = model.evaluate(testX, testY, verbose=0, return_dict=True)
    print('Test loss:', score['loss'])
    print('Test accuracy:', score['accuracy'])
    print('Test AUC:', score['auc'])

if __name__ == '__main__':
    # shallow()
    deep()
