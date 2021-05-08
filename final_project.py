import numpy as np
import pandas as pd
from os import listdir, path
from PIL import Image

# creates positive samples from the data set
def save_positive_samples(df, i):
    trainData = [list(), list()]

    for index, row in df.iterrows():
        
        try:
            # get all images in the 2 directories
            imgs_p1 = listdir(row['p1'])
            imgs_p2 = listdir(row['p2'])

            # all combinations of the images in either directory
            for file1 in imgs_p1:

                img1 = Image.open(row['p1']+'/'+file1)
                for file2 in imgs_p2:
                    img2 = Image.open(row['p2']+'/'+file2)

                    # resize image and turn into an np array
                    trainData[0].append(np.asarray(img1.resize((128, 128), Image.ANTIALIAS)))
                    trainData[1].append(np.asarray(img2.resize((128, 128), Image.ANTIALIAS)))
        except FileNotFoundError:
            pass

    # save all the images in a numpy file
    data = np.array(trainData)
    np.save("training_images_pos{}.npy".format(i), data)

# creates negative samples from the data set
def save_negative_samples(df, i):
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

                img1 = Image.open(row['p1']+'/'+file1)
                for file2 in imgs_p2:
                    img2 = Image.open(row['p2']+'/'+file2)

                    # resize image and turn into an np array
                    person1.append(np.asarray(img1.resize((128, 128), Image.ANTIALIAS)))
                    person2.append(np.asarray(img2.resize((128, 128), Image.ANTIALIAS)))
        except FileNotFoundError:
            pass

    np.random.shuffle(person2)

    trainData.append(person1)
    trainData.append(person2)
    # save all the images in a numpy file
    data = np.array(trainData)
    np.save("training_images_neg{}.npy".format(i), data)

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

    # random sample of 20% of the data set
    p = .05
    for i in range(1):
        b = df.sample(frac=p)
        save_positive_samples(b, i)

    for i in range(1):
        b = df.sample(frac=p)
        save_negative_samples(b, i)

if __name__ == '__main__':
    if(not path.exists('training_images_pos0.npy') and 
        not path.exists('training_image_neg0.npy')):
        convert_to_numpy()
    else:
        train_pos = np.load('training_images_pos0.npy')
        train_neg = np.load('training_images_neg0.npy')

        print(train_pos.shape)
        print(train_neg.shape)
        pass
    pass