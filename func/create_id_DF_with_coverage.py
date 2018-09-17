import pandas as pd
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

# Loading of training/testing ids and depths
train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = df.id.apply(lambda x:cv_resize(read_img(f'../data/train/{x}.png')
train_df["masks"] = [np.array(load_img("../data/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(101, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

train_df = train_df.drop('masks', axis=1)
train_df = train_df.drop('images', axis=1)

train_df.to_csv('./train_df_coverage.csv')
test_df.to_csv('./test_df.csv')

