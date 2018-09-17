
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time, sys, json, os, io, cv2, base64
from io import BytesIO
from subprocess import check_output
from pprint import pprint as pp
import pymongo
from pymongo import MongoClient
import hashlib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from random import randint
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, BatchNormalization
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D, Concatenate, SpatialDropout2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers.merge import add
from keras import regularizers
from keras.regularizers import l2
import tensorflow as tf

from tqdm import tqdm_notebook


# In[2]:


path_train = './data/train/'
path_test = './data/test/'
train_ids = next(os.walk(path_train))[2]
test_ids = next(os.walk(path_test))[2]
depths_df = pd.read_csv("./data/depths.csv", index_col="id")
train_df = pd.read_csv("./data/train.csv", index_col="id")


# In[3]:


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """
    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)
    return conn[db]

def insert_data(data, db, collection, check_id='id', host='localhost', port=27017, username=None, password=None, no_id=True):
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    
    duplicate_result = db[collection].find(
       {check_id: data[check_id]})
    
    print('duplicate count' + str(duplicate_result.count()))
    if duplicate_result.count() == 0:
        db[collection].insert_one(data)

def set_data(db, collection, search, set_query, host='localhost', port=27017, username=None, password=None, no_id=True):
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    db[collection].update(
        search,
        { '$set': set_query }
    )
        
def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df

def sample_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True, num_sample=1000):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].aggregate([{ "$sample": { "size": num_sample }}])

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df


# In[19]:


img_size_ori = 101
img_size_target = 128

def read_resize_img(x, scale):
    img_byte = io.BytesIO(base64.b64decode(x))
    img_np = np.array(imread(img_byte, as_gray=True))/scale
    img_resize = resize(img_np,(img_size_target,img_size_target), mode='constant', preserve_range=True)
    return img_resize

# train_df = read_mongo('dataset', 'tgs_salt', {"$and": [{"img_mask_base64":{"$ne":None}}, {"contrast":{"$ne":None}}]})
train_df = read_mongo('dataset', 'tgs_salt', {"$and": [{"img_mask_base64":{"$ne":None}}]})
train_df['img'] = train_df['img_base64'].apply(lambda x: read_resize_img(x, 256.0))
train_df['img_mask'] = train_df['img_mask_base64'].apply(lambda x: read_resize_img(x, 65535.0))
# train_df['img_contrast'] = train_df['contrast'].apply(lambda x: read_resize_img(x, 256.0))
# train_df['img_correlation'] = train_df['correlation'].apply(lambda x: read_resize_img(x, 256.0))
# train_df['img_dissimilarity'] = train_df['dissimilarity'].apply(lambda x: read_resize_img(x, 256.0))
# train_df['img_energy'] = train_df['energy'].apply(lambda x: read_resize_img(x, 256.0))
# train_df['img_entropy'] = train_df['entropy'].apply(lambda x: read_resize_img(x, 256.0))
# train_df['img_homogeneity'] = train_df['homogeneity'].apply(lambda x: read_resize_img(x, 256.0))
# train_df['img_ASM'] = train_df['ASM'].apply(lambda x: read_resize_img(x, 256.0))

train_df = train_df.drop('img_base64', axis=1)
train_df = train_df.drop('img_mask_base64', axis=1)
# train_df = train_df.drop('contrast', axis=1)
# train_df = train_df.drop('correlation', axis=1)
# train_df = train_df.drop('dissimilarity', axis=1)
# train_df = train_df.drop('energy', axis=1)
# train_df = train_df.drop('entropy', axis=1)
# train_df = train_df.drop('homogeneity', axis=1)
# train_df = train_df.drop('ASM', axis=1)
    
train_df, val_df = train_test_split(train_df, test_size=0.1)

# train_df.head()


# In[20]:


def random_crop_resize(x, crop, flip):
    if flip:
        x = np.fliplr(x)
    x = x[crop[0]:-crop[1],crop[2]:-crop[3]]
    x = resize(x,(img_size_target,img_size_target), mode='constant', preserve_range=True)
    return x

def img_augment(df):
    augment_df = pd.DataFrame(columns=df.columns)
    for index, row in tqdm_notebook(df.iterrows(),total=len(df.index)):
    #     np.random.seed(0)
        crop = np.random.randint(low=1, high=10, size=4)
        flip = np.random.choice([True, False])

        img = random_crop_resize(row['img'], crop, flip)
        img_mask = random_crop_resize(row['img_mask'], crop, flip)
    #     img_contrast = random_crop_resize(row['img_contrast'], crop, flip)
    #     img_correlation = random_crop_resize(row['img_correlation'], crop, flip)
    #     img_dissimilarity = random_crop_resize(row['img_dissimilarity'], crop, flip)
    #     img_energy = random_crop_resize(row['img_energy'], crop, flip)
    #     img_entropy = random_crop_resize(row['img_entropy'], crop, flip)
    #     img_homogeneity = random_crop_resize(row['img_homogeneity'], crop, flip)
    #     img_ASM = random_crop_resize(row['img_ASM'], crop, flip)
        coverage = np.sum(img_mask) / pow(img_size_target, 2)
        coverage_class = np.ceil(coverage*10).astype(np.int_)

        augment_df = augment_df.append(
            {
                'coverage': coverage,
                'coverage_class': coverage_class,
                'depth': row['depth'],
                'img_id': row['img_id']+'_augment',
                'img_size': row['img_size'],
                'img': img,
                'img_mask': img_mask,
    #             'img_contrast': img_contrast,
    #             'img_correlation': img_correlation,
    #             'img_dissimilarity': img_dissimilarity,
    #             'img_energy': img_energy,
    #             'img_entropy': img_entropy,
    #             'img_homogeneity': img_homogeneity,
    #             'img_ASM': img_ASM
            }, ignore_index=True
        )

    all_df = pd.concat([df, augment_df],ignore_index=True)
    return all_df

train_df = img_augment(train_df)
val_df = img_augment(val_df)

base_idx = 600
max_images = 32
grid_width = 8
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(20, 20))
row = 0; col = 0;
for i, idx in enumerate(train_df.index[base_idx:base_idx+grid_height]):
    print(f'shape: {train_df.loc[idx].img.shape}, id: {train_df.loc[idx].img_id}, depth: {train_df.loc[idx].depth}, coverage: {train_df.loc[idx].coverage}, coverage_class: {train_df.loc[idx].coverage_class}')
    img = train_df.loc[idx].img
    img_mask = train_df.loc[idx].img_mask
    img_contrast = train_df.loc[idx].img_contrast
    img_correlation = train_df.loc[idx].img_correlation
    img_dissimilarity = train_df.loc[idx].img_dissimilarity
    img_energy = train_df.loc[idx].img_energy
    img_entropy = train_df.loc[idx].img_entropy
    img_homogeneity = train_df.loc[idx].img_homogeneity
    img_ASM = train_df.loc[idx].img_ASM
    
    ax = axs[row, col]; ax.imshow(img, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]); col+=1;
    ax = axs[row, col]; ax.imshow(img_contrast, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]);col+=1;
    ax = axs[row, col]; ax.imshow(img_correlation, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]);col+=1;
    ax = axs[row, col]; ax.imshow(img_dissimilarity, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]);col+=1;
    ax = axs[row, col]; ax.imshow(img_energy, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]);col+=1;
    ax = axs[row, col]; ax.imshow(img_entropy, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]);col+=1;
    ax = axs[row, col]; ax.imshow(img_homogeneity, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]);col+=1;
    ax = axs[row, col]; ax.imshow(img_ASM, cmap="seismic"); ax.imshow(img_mask, alpha=0.3, cmap="Reds"); ax.set_yticklabels([]); ax.set_xticklabels([]);col=0; row+=1;# img_stack = np.stack((np.asarray(train_df['img'].values.tolist()),
#                       np.asarray(train_df['img_contrast'].values.tolist()),
#                       np.asarray(train_df['img_correlation'].values.tolist()),
#                       np.asarray(train_df['img_dissimilarity'].values.tolist()),
#                       np.asarray(train_df['img_energy'].values.tolist()),
#                       np.asarray(train_df['img_entropy'].values.tolist()),
#                       np.asarray(train_df['img_homogeneity'].values.tolist()),
#                       np.asarray(train_df['img_ASM'].values.tolist())))
# # img_stack = np.stack((np.asarray(train_df['img'].values.tolist()),
# #                       np.asarray(train_df['img_dissimilarity'].values.tolist()),
# #                       np.asarray(train_df['img_entropy'].values.tolist())
# #                       ))
# img_stack = np.transpose(img_stack,(1,2,3,0))

img_stack = np.stack((np.asarray(train_df['img'].values.tolist())))
img_stack = np.expand_dims(img_stack,axis=3)

# for later test normalization
depth_mean = np.mean(train_df['depth'].values)
depth_std = np.std(train_df['depth'].values)

# train_X = (img_stack - np.mean(img_stack,axis=3,keepdims=True)) / np.std(img_stack,axis=3,keepdims=True)
train_X = img_stack
train_d = (train_df['depth'].values-depth_mean) / depth_std
train_y = np.expand_dims(np.asarray(train_df['img_mask'].values.tolist()),axis=3)

# Split train and valid
X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(train_X, train_d, train_y, test_size=0.1, random_state=42)
# In[25]:


X_train = np.expand_dims(np.stack((np.asarray(train_df['img'].values.tolist()))),axis=3)
X_valid = np.expand_dims(np.stack((np.asarray(val_df['img'].values.tolist()))),axis=3)
y_train = np.expand_dims(np.asarray(train_df['img_mask'].values.tolist()),axis=3)
y_valid = np.expand_dims(np.asarray(val_df['img_mask'].values.tolist()),axis=3)


# In[22]:


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# tensorflow session setting
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

im_width = 128
im_height = 128
border = 5
im_chan = 1 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth
neurons_num = 16

# Build U-Net model
input_img = Input((im_height, im_width, im_chan), name='img')
input_features = Input((n_features, ), name='feat')

c1 = Conv2D(neurons_num*1, (3, 3), activation='relu', padding='same') (input_img)
c1 = BatchNormalization()(c1)
c1 = Conv2D(neurons_num*1, (3, 3), activation='relu', padding='same') (c1)
c1 = BatchNormalization()(c1)
sc_1 = Conv2D(neurons_num*1, (1, 1), padding='same') (input_img)
c1 = add([c1, sc_1])
c1 = BatchNormalization()(c1)
p1 = MaxPooling2D((2, 2)) (c1)
p1 = Dropout(0.25)(p1)

c2 = Conv2D(neurons_num*2, (3, 3), activation='relu', padding='same') (p1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(neurons_num*2, (3, 3), activation='relu', padding='same') (c2)
c2 = BatchNormalization()(c2)
sc_2 = Conv2D(neurons_num*2, (1, 1), padding='same') (p1)
c2 = add([c2, sc_2])
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)
p2 = Dropout(0.5)(p2)

c3 = Conv2D(neurons_num*4, (3, 3), activation='relu', padding='same') (p2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(neurons_num*4, (3, 3), activation='relu', padding='same') (c3)
c3 = BatchNormalization()(c3)
sc_3 = Conv2D(neurons_num*4, (1, 1), padding='same') (p2)
c3 = add([c3, sc_3])
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2)) (c3)
p3 = Dropout(0.5)(p3)

c4 = Conv2D(neurons_num*8, (3, 3), activation='relu', padding='same') (p3)
c4 = BatchNormalization()(c4)
c4 = Conv2D(neurons_num*8, (3, 3), activation='relu', padding='same') (c4)
c4 = BatchNormalization()(c4)
sc_4 = Conv2D(neurons_num*8, (1, 1), padding='same') (p3)
c4 = add([c4, sc_4])
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
p4 = Dropout(0.5)(p4)

# Join features information in the depthest layer
f_repeat = RepeatVector(8*8)(input_features)
f_conv = Reshape((8, 8, n_features))(f_repeat)
p4_feat = concatenate([p4, f_conv], -1)

c5 = Conv2D(neurons_num*16, (3, 3), activation='relu', padding='same') (p4_feat)
c5 = BatchNormalization()(c5)
c5 = Conv2D(neurons_num*16, (3, 3), activation='relu', padding='same') (c5)
c5 = BatchNormalization()(c5)
sc_5 = Conv2D(neurons_num*16, (1, 1), padding='same') (p4_feat)
c5 = add([c5, sc_5])
c5 = BatchNormalization()(c5)

u6 = Conv2DTranspose(neurons_num*8, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
u6 = Dropout(0.5)(u6)
c6 = Conv2D(neurons_num*8, (3, 3), activation='relu', padding='same') (u6)
c6 = BatchNormalization()(c6)
c6 = Conv2D(neurons_num*8, (3, 3), activation='relu', padding='same') (c6)
c6 = BatchNormalization()(c6)
sc_6 = Conv2D(neurons_num*8, (1, 1), padding='same') (u6)
c6 = add([c6, sc_6])
c6 = BatchNormalization()(c6)

u7 = Conv2DTranspose(neurons_num*4, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
u7 = Dropout(0.5)(u7)
c7 = Conv2D(neurons_num*4, (3, 3), activation='relu', padding='same') (u7)
c7 = BatchNormalization()(c7)
c7 = Conv2D(neurons_num*4, (3, 3), activation='relu', padding='same') (c7)
c7 = BatchNormalization()(c7)
sc_7 = Conv2D(neurons_num*4, (1, 1), padding='same') (u7)
c7 = add([c7, sc_7])
c7 = BatchNormalization()(c7)

u8 = Conv2DTranspose(neurons_num*2, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
u8 = Dropout(0.5)(u8)
c8 = Conv2D(neurons_num*2, (3, 3), activation='relu', padding='same') (u8)
c8 = BatchNormalization()(c8)
c8 = Conv2D(neurons_num*2, (3, 3), activation='relu', padding='same') (c8)
c8 = BatchNormalization()(c8)
sc_8 = Conv2D(neurons_num*2, (1, 1), padding='same') (u8)
c8 = add([c8, sc_8])
c8 = BatchNormalization()(c8)

u9 = Conv2DTranspose(neurons_num*1, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
u9 = Dropout(0.5)(u9)
c9 = Conv2D(neurons_num*1, (3, 3), activation='relu', padding='same') (u9)
c9 = BatchNormalization()(c9)
c9 = Conv2D(neurons_num*1, (3, 3), activation='relu', padding='same') (c9)
c9 = BatchNormalization()(c9)
sc_9 = Conv2D(neurons_num*1, (1, 1), padding='same') (u9)
c9 = add([c9, sc_9])
c9 = BatchNormalization()(c9)

# c9 = Dropout(0.5)(c9)
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[input_img, input_features], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou, 'accuracy']) #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...
model.summary()
# In[23]:


def conv_block(m, dim, acti, bn, res, do=0, l2_lamda=0):
	n = Conv2D(dim, 3, activation=acti, padding='same',kernel_regularizer=l2(l2_lamda),bias_regularizer=l2(l2_lamda),activity_regularizer=l2(l2_lamda))(m)
	n = BatchNormalization(beta_regularizer=l2(l2_lamda), gamma_regularizer=l2(l2_lamda))(n) if bn else n
	n = SpatialDropout2D(do/2.0)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, padding='same',kernel_regularizer=l2(l2_lamda),bias_regularizer=l2(l2_lamda),activity_regularizer=l2(l2_lamda))(n)
	n = BatchNormalization(beta_regularizer=l2(l2_lamda), gamma_regularizer=l2(l2_lamda))(n) if bn else n
	n = Concatenate()([m, n]) if res else n
	n = SpatialDropout2D(do)(n) if do else n
	return n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res, l2_lamda):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res, l2_lamda)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same', kernel_regularizer=l2(l2_lamda),bias_regularizer=l2(l2_lamda),activity_regularizer=l2(l2_lamda))(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same', kernel_regularizer=l2(l2_lamda),bias_regularizer=l2(l2_lamda),activity_regularizer=l2(l2_lamda))(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res, l2_lamda=l2_lamda)
	else:
		m = conv_block(m, dim, acti, bn, res, do, l2_lamda=l2_lamda)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=True, l2_lamda=0):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual, l2_lamda)
	o = Conv2D(out_ch, 1, activation='sigmoid', kernel_regularizer=l2(l2_lamda),bias_regularizer=l2(l2_lamda),activity_regularizer=l2(l2_lamda))(o)
	return Model(inputs=i, outputs=o)

model = UNet((img_size_target,img_size_target,1),start_ch=16,depth=5,batchnorm=True, dropout=0.8, l2_lamda=0.0)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[mean_iou,"accuracy"])
model.summary()


# In[26]:


epochs = 30
batch_size = 32
callbacks = [
    EarlyStopping(patience=10, verbose=1, monitor="val_mean_iou", mode="max"),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-unet-resnet.h5', verbose=1, save_best_only=True, monitor="val_mean_iou", mode="max")
]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))

# history = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
#                     validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))


# In[27]:


fig, (ax_loss, ax_acc, ax_iou) = plt.subplots(1, 3, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_acc.plot(history.epoch, history.history["acc"], label="Train iou")
ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation iou")
ax_iou.plot(history.epoch, history.history["mean_iou"], label="Train iou")
ax_iou.plot(history.epoch, history.history["val_mean_iou"], label="Validation iou")


# In[31]:


# model = load_model("./model-unet-resnet.h5", custom_objects={'mean_iou':mean_iou})
# preds_valid = model.predict({'img': train_X, 'feat': train_d}, batch_size=32, verbose=1)
# preds_valid = model.predict(train_X, batch_size=32, verbose=1)
preds_valid = model.predict(X_valid, batch_size=32, verbose=1)


# In[32]:


base_idx = 351
max_images = 32
grid_width = 4
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(20, 20))
row = 0; col = 0;
for i, idx in enumerate(train_df.index[base_idx:base_idx+int(max_images/2)]):
    img = val_df.loc[idx].img
    mask = val_df.loc[idx].img_mask
    pred = preds_valid[idx].squeeze()
    
    ax = axs[row, col];
    ax.imshow(img, cmap="seismic")
    ax.imshow(mask, alpha=0.8, cmap="Reds"); col+=1;
    ax.set_yticklabels([]); ax.set_xticklabels([]);
    
    ax = axs[row, col];
    ax.imshow(img, cmap="seismic")
    ax.imshow(pred, alpha=0.8, cmap="Reds"); col+=1;
    ax.set_yticklabels([]);ax.set_xticklabels([]);
    
    if col >= grid_width:
        col=0; row+=1;


# In[35]:


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch],print_table=False)
        metric.append(value)
    return np.mean(metric)


from math import sqrt
from joblib import Parallel, delayed
import multiprocessing  
from tqdm import tqdm  

thresholds = np.linspace(0, 1, 20)
# result = Parallel(n_jobs=2)(io_metric_batch(train_y, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds))
# ious = np.array([iou_metric_batch(train_y, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])
ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])


# In[36]:


threshold_best_index = np.argmax(ious[3:-3])
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

print(f'threshold_best: {threshold_best}, iou_best: {iou_best}')

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

