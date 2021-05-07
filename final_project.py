import numpy as np
import pandas as pd
from os import listdir, path
from PIL import Image

def save_batch(df, i):
    trainData = [list(), list()]
    for index, row in df.iterrows():
        
        try:
            imgs_p1 = listdir(row['p1'])
            imgs_p2 = listdir(row['p2'])

            for file1 in imgs_p1:

                img1 = Image.open(row['p1']+'/'+file1)
                for file2 in imgs_p2:
                    img2 = Image.open(row['p2']+'/'+file2)

                    trainData[0].append(np.asarray(img1))
                    trainData[1].append(np.asarray(img2))
        except FileNotFoundError:
            pass

    # save all the images in a numpy file
    data = np.array(trainData)
    np.save("training_images{}.npy".format(i), data)
    pass

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
    for i in range(20):
        b = df.sample(frac=p)
        save_batch(b, i)

if __name__ == '__main__':
    if(not path.exists('training_images.npy')):
        convert_to_numpy()
    pass