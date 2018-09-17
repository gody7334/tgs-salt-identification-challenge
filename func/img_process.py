import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from skimage.transform import resize

# img_size_ori = 101
# img_size_target = 101

def read_img(path):
    return np.array(load_img(path, grayscale=True)) / 255

def create_train_val_test_split_df():
    train_df = pd.read_csv("../func/train_df_coverage.csv")
    test_df = pd.read_csv("../func/test_df.csv")
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)
    return train_df, val_df, test_df

def read_train_img_to_np_array(df):
    return np.stack((np.array(df.id.apply(lambda x:cv_resize(read_img(f'../data/train/{x}.png'), 128)).tolist()),)*3, -1)
  
def read_test_img_to_np_array(df):
    return np.stack((np.array(df.id.apply(lambda x:cv_resize(read_img(f'../data/test/{x}.png'), 128)).tolist()),)*3, -1)
    
def read_mask_img_to_np_array(df):
    return np.array(df.id.apply(lambda x:cv_resize(read_img(f'../data/masks/{x}.png'), 128)).tolist()).reshape(-1, 128, 128, 1)

# inter=cv2.INTER_LINEAR
inter=cv2.INTER_NEAREST

def cv_resize(img, img_size_target):# not used
    return cv2.resize(img, dsize=(img_size_target, img_size_target), interpolation=inter)

def upsample(img, img_size_target):# not used
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img, img_size_ori):# not used
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def uppad(img):# not used
    return np.pad(img, ((14, 13), (14, 13)), 'reflect')

def downpad(img):# not used
    return img[14:-13, 14:-13]