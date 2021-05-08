import numpy as np
import pandas as pd
from os import listdir, path
from PIL import Image, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as lr
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.utils._testing import ignore_warnings


# creates positive samples from the data set
def save_positive_samples(df, i, type):
    trainData = []

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

                    # sample = []

                    # resize image and turn into an np array
                    # sample.append(np.asarray(img1.resize((128, 128), Image.ANTIALIAS)))
                    # sample.append(np.asarray(img2.resize((128, 128), Image.ANTIALIAS)))

                    trainData.append(
                        np.concatenate(
                            (np.asarray(ImageOps.grayscale(img1.resize((128, 128), Image.ANTIALIAS))),
                             np.asarray(ImageOps.grayscale(img2.resize((128, 128), Image.ANTIALIAS))))))

        except FileNotFoundError:
            pass

    # save all the images in a numpy file
    data = np.array(trainData)
    np.save("{}_images_pos{}.npy".format(type, i), data)


# creates negative samples from the data set
def save_negative_samples(df, i, type):
    trainData = []
    person1 = []
    person2 = []

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

                    # resize image and turn into an np array
                    person1.append(np.asarray(ImageOps.grayscale(img1.resize((128, 128), Image.ANTIALIAS))))
                    person2.append(np.asarray(ImageOps.grayscale(img2.resize((128, 128), Image.ANTIALIAS))))
        except FileNotFoundError:
            pass

    np.random.shuffle(person2)

    # trainData.append(person1)
    # trainData.append(person2)
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
    # print(df)

    # iterate over everything in the csv
    # for index, row in df.iterrows():
    #     print(row['p1'], row['p2'])

    # set up correct path to images
    df['p1'] = 'data/train/' + df['p1']
    df['p2'] = 'data/train/' + df['p2']

    # random sample of 5% of the data set
    p = .05
    for i in range(1):
        b = df.sample(frac=p)
        save_positive_samples(b, i, "training")

    for i in range(1):
        b = df.sample(frac=p)
        save_negative_samples(b, i, 'training')

    for i in range(1):
        b = df.sample(frac=p)
        save_positive_samples(b, i, "testing")

    for i in range(1):
        b = df.sample(frac=p)
        save_negative_samples(b, i, 'testing')


@ignore_warnings(category=ConvergenceWarning)
def main():
    if (not path.exists('training_images_pos0.npy') and
            not path.exists('training_image_neg0.npy')):
        convert_to_numpy()

        # (# samples, 256, 128, 3)
    train_pos = np.load('training_images_pos0.npy')
    train_neg = np.load('training_images_neg0.npy')
    print(train_pos.shape)
    print(train_neg.shape)

    # flatten images
    train_pos = preprocessing.minmax_scale(train_pos.reshape((train_pos.shape[0], 32768)))
    train_neg = preprocessing.minmax_scale(train_neg.reshape((train_neg.shape[0], 32768)))

    pos_label = np.ones(train_pos.shape[0])
    neg_label = np.zeros(train_neg.shape[0])

    trainX = np.concatenate((train_pos, train_neg))
    trainY = np.concatenate((pos_label, neg_label))

    idxs = np.random.permutation(trainX.shape[0])

    trainX = trainX[idxs]
    trainY = trainY[idxs]

    # train_pos = np.concatenate((train_pos, pos_label), axis=1)
    # train_neg = np.concatenate((train_neg, neg_label), axis=1)

    clf = lr().fit(trainX, trainY)

    test_pos = np.load('testing_images_pos0.npy')
    test_neg = np.load('testing_images_neg0.npy')

    test_pos = preprocessing.minmax_scale(test_pos.reshape((test_pos.shape[0], 32768)))
    test_neg = preprocessing.minmax_scale(test_neg.reshape((test_neg.shape[0], 32768)))

    pos_label_test = np.ones(test_pos.shape[0])
    neg_label_test = np.zeros(test_neg.shape[0])

    testX = np.concatenate((test_pos, test_neg))
    testY = np.concatenate((pos_label_test, neg_label_test))

    idxs = np.random.permutation(testX.shape[0])

    testX = testX[idxs]
    testY = testY[idxs]

    print(clf.score(testX, testY))

    # plt.imshow(train_pos[0])
    # plt.show()


if __name__ == '__main__':
    main()
