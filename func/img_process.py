import numpy as np
import pandas as pd
import skimage as sk
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from skimage.transform import resize

def _convert_to_np_array(_augment_df):
    X_np = np.stack((np.stack((np.asarray(_augment_df['aug_img'].values.tolist()))),)*3,-1)
    y_np = np.expand_dims(np.asarray(_augment_df['aug_img_mask'].values.tolist()),axis=3)
    
    return X_np, y_np

img_size_ori = 101
img_size_target = 128

def _img_augmentation(df, if_augment=True, target_size=128):
    augment_df = pd.DataFrame()
    for index, row in df.iterrows():
        # np.random.seed(0)
        crop = np.random.randint(low=1, high=20, size=4) if if_augment else np.array([0, 1, 0, 1])
        flip = np.random.choice([True, False]) if if_augment else False
        degree = np.random.uniform(-10, 10) if if_augment else 0

        aug_img = random_crop_resize(row['img'], crop, flip, degree, target_size=target_size)
        aug_img_mask = random_crop_resize(row['img_mask'], crop, flip, degree, target_size=target_size)

        augment_df = augment_df.append(
            {
                'img_id': row['id']+'_augment',
                'aug_img': aug_img,
                'aug_img_mask': aug_img_mask,
            }, ignore_index=True
        )

    return augment_df

def random_crop_resize(x, crop, flip, degree, noise_type='None', target_size=128):
    ######### noise, rotate decrease the performance a lot... ###########
#         x = np.squeeze(noisy(noise_type, np.expand_dims(x, axis=3)))

#     x = sk.transform.rotate(x, degree, mode='reflect')
    x = np.fliplr(x) if flip else x
    x = x[crop[0]:-crop[1],crop[2]:-crop[3]]
    x = pad_resize(x, target_size)

    return x

def pad_resize(x, target_size=128, mode='reflect'):
    # resize image using pad reflect, it has the best performance so far
    w,h = x.shape
    pad1 = int((target_size-w)/2)
    pad2 = target_size - (w+pad1)
    pad3 = int((target_size-h)/2)
    pad4 = target_size - (h+pad3)
    x = np.pad(x, ((pad1, pad2), (pad3, pad4)), mode)
    return x

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.0005
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.01
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      scale = 0.05
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss * scale
      return noisy
   elif noise_typ == 'None':
      return image

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

def cv_resize(img, img_size_target):# not used
    # inter=cv2.INTER_LINEAR
    inter=cv2.INTER_NEAREST
    return cv2.resize(img, dsize=(img_size_target, img_size_target), interpolation=inter)

def upsample(img, img_size_target):# not used
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img, img_size_ori):# not used
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def uppad(img):# not used
    return np.pad(img, ((14, 13), (14, 13)), 'reflect')

def downpad(img):# not used
    return img[14:-13, 14:-13]

