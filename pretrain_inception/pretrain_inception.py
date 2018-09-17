
# coding: utf-8

# ### U-net with simple Resnet Blocks v2, can get 0.80+
# * Original version : 
#   https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks
#         
#         
# #### update log
# 1.   Cancel last dropout (seems better)
# 2.  modify convolution_block, to be more consistant with the standard resent model. 
#       * https://arxiv.org/abs/1603.05027
# 3. Use faster  IOU metric score code,
#       * https://www.kaggle.com/donchuk/fast-implementation-of-scoring-metric
# 4. Use  binary_crossentropy loss and then Lovász-hinge loss (very slow!)
#      * Lovász-hinge loss: https://github.com/bermanmaxim/LovaszSoftmax
#      
# Limit the max epochs number to make the kernel finish in the limit of 6 hours, better score can be achived at more epochs 

# In[1]:


# notebook lib
import pydot
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook #, tnrange
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import time
t_start = time.time()


# In[16]:


import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from func import img_process
from func import custom_loss
from func import post_process
from func.DataGenerator import DataGenerator

import pandas as pd
import numpy as np

import tensorflow as tf
from segmentation_models import Unet
from keras.backend import tensorflow_backend, common
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model, save_model
from keras import backend as K
from keras import optimizers


# In[3]:


version = 5
basic_name = f'Unet_resnet_v{version}'
save_model_name = basic_name + '.model'
submission_file = basic_name + '.csv'

print(save_model_name)
print(submission_file)


# In[4]:


# keep split same
if(os.path.isfile('val_df.csv') is False):
    train_df, val_df, test_df = img_process.create_train_val_test_split_df()
    train_df.to_csv('./train_df.csv')
    val_df.to_csv('./val_df.csv')
    test_df.to_csv('./test_df.csv')
else:
    train_df = pd.read_csv("./train_df.csv")
    val_df = pd.read_csv("./val_df.csv")
    test_df = pd.read_csv("./test_df.csv")


# In[5]:


# tensorflow session setting
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))


# In[6]:


# prepare model
model1 = Unet(backbone_name='inceptionresnetv2', encoder_weights='imagenet')
c = optimizers.adam(0.001)
model1.compile(loss= custom_loss.bce_dice_loss, optimizer=c, metrics=[custom_loss.my_iou_metric])

# model1.summary()
# SVG(model_to_dot(model1, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))


# In[7]:


epochs = 1
batch_size = 16

#early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

training_generator = DataGenerator( train_df, batch_size=batch_size)
validation_generator = DataGenerator( val_df, batch_size=batch_size)

history = model1.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs, callbacks=[model_checkpoint,reduce_lr],
                    verbose=2,      
                    use_multiprocessing=True,
                    workers=4)

# history = model1.fit(x_train, y_train,
#                     validation_data=[x_valid, y_valid], 
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     callbacks=[ model_checkpoint,reduce_lr], 
#                     verbose=2)


# In[9]:


model1 = load_model(save_model_name,custom_objects={'bce_dice_loss': custom_loss.bce_dice_loss, 'my_iou_metric': custom_loss.my_iou_metric})
# model1 = load_model(save_model_name,custom_objects={'my_iou_metric': my_iou_metric, 'lovasz_loss': lovasz_loss, 'my_iou_metric_2': my_iou_metric_2})
# remove layter activation layer and use losvasz loss
input_x = model1.layers[0].input

output_layer = model1.layers[-1].input
model = Model(input_x, output_layer)
c = optimizers.adam(0.0005)

# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model.compile(loss=custom_loss.lovasz_loss, optimizer=c, metrics=[custom_loss.my_iou_metric_2])

#model.summary()


# In[10]:


early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.00001, verbose=1)
epochs = 2
batch_size = 16

training_generator = DataGenerator( train_df, batch_size=batch_size)
validation_generator = DataGenerator( val_df, batch_size=batch_size)

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs, callbacks=[model_checkpoint,reduce_lr, early_stopping],
                    verbose=2,      
                    use_multiprocessing=True,
                    workers=4)

# history = model.fit(x_train, y_train,
#                     validation_data=[x_valid, y_valid], 
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
#                     verbose=2)


# In[14]:


fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_score.plot(history.epoch, history.history["my_iou_metric_2"], label="Train score")
ax_score.plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation score")
ax_score.legend()
savefig('loss_score.png')


# In[7]:


model = load_model(save_model_name,custom_objects={'my_iou_metric_2': custom_loss.my_iou_metric_2,
                                                   'lovasz_loss': custom_loss.lovasz_loss})


# In[8]:


x_valid = img_process.read_train_img_to_np_array(val_df)
y_valid = img_process.read_mask_img_to_np_array(val_df)


# In[9]:


def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2


# In[10]:


preds_valid = predict_result(model,x_valid,128)


# In[12]:


y_valid_pad = y_valid[:,14:-13, 14:-13,:]
preds_valid_pad = preds_valid[:,14:-13, 14:-13]


# In[15]:


## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

# ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# print(ious)
ious = np.array([custom_loss.iou_metric_batch(y_valid_pad, preds_valid_pad > threshold) for threshold in tqdm_notebook(thresholds)])
print(ious)


# In[14]:


# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()


# In[11]:


"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[16]:


x_test = np.array([uppad((np.array(load_img("../data/test/{}.png".format(idx), grayscale = True))) / 255) for idx in tqdm_notebook(test_df.index)])
x_test_stack = np.stack((x_test,)*3,-1)


# In[17]:


x_test_stack.shape


# In[18]:


preds_test = predict_result(model,x_test_stack,128)


# In[21]:


preds_test = preds_test[:,14:-13, 14:-13]


# In[23]:


threshold_best = 0.32277339
t1 = time.time()
pred_dict = {idx: rle_encode(np.round(preds_test[i] > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
t2 = time.time()

print(f"Usedtime = {t2-t1} s")


# In[24]:


sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)


# In[ ]:


t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")

