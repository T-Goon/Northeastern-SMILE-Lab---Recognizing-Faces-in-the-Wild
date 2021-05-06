import numpy as np
import pandas as pd
from os import listdir
from PIL import Image

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
    # print(df.sample(frac=.2))

    trainData = [list(), list()]
    for index, row in df.iterrows():
        # print(row['p1'], row['p2'])
        try:
            imgs_p1 = listdir(row['p1'])
            imgs_p2 = listdir(row['p2'])

            for file1, file2 in zip(imgs_p1, imgs_p2):

                img1 = Image.open(row['p1']+'/'+file1)
                img2 = Image.open(row['p2']+'/'+file2)

                trainData[0].append(np.asarray(img1))
                trainData[1].append(np.asarray(img2))
        except FileNotFoundError:
            pass

    # save all the images in a numpy file
    data = np.array(trainData)

    np.save("training_images.npy", data)

if __name__ == '__main__':
#    convert_to_numpy()
    pass