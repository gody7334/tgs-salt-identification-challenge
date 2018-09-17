''' expeirment discription 
model: Unet with inceptionV2 backbone
loss: bce+dice loss -> lovazs loss
augment: flip, crop, pad reflect resize
epoch: 100 -> 200
lr: 0.001 -> 0.0005

Setup experiment pipeline, Try to reproduce the best score = 0.82
'''
TAG = '000_INCEPTION_UNET'

basic_name = f'{TAG}'
save_model_name = basic_name + '.model'
submission_file = basic_name + '.csv'

print(save_model_name)
print(submission_file)

device = '0'
first_epoch = 100
second_epoch = 200
batch_size = 16
first_lr = 0.001
second_lr = 0.0005

#############################################################
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
os.environ["CUDA_VISIBLE_DEVICES"]=device

from func import img_process
from func import custom_loss
from func.DataGenerator import DataGenerator

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

import tensorflow as tf
from segmentation_models import Unet
from keras.backend import tensorflow_backend, common
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model, save_model
from keras import backend as K
from keras import optimizers
#############################################################

print('load dataset')
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

print('set tf session')
# tensorflow session setting
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

print('prepare model')
model1 = Unet(backbone_name='inceptionresnetv2', encoder_weights='imagenet')
c = optimizers.adam(first_lr)
model1.compile(loss= custom_loss.bce_dice_loss, optimizer=c, metrics=[custom_loss.my_iou_metric])

print('first training')
epochs = first_epoch
batch_size = batch_size

#early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric', mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

training_generator = DataGenerator( train_df, batch_size=batch_size)
validation_generator = DataGenerator( val_df, batch_size=batch_size)

history = model1.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=[model_checkpoint,reduce_lr],
                    verbose=2,
                    use_multiprocessing=True,
                    workers=4)

print('load first model')
model2 = load_model(save_model_name,custom_objects={'bce_dice_loss': custom_loss.bce_dice_loss, 'my_iou_metric': custom_loss.my_iou_metric})
input_x = model2.layers[0].input

output_layer = model2.layers[-1].input
model2 = Model(input_x, output_layer)
c = optimizers.adam(second_lr)

# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model2.compile(loss=custom_loss.lovasz_loss, optimizer=c, metrics=[custom_loss.my_iou_metric_2])

print('second training')
epochs = second_epoch
batch_size = batch_size

early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.00001, verbose=1)

history = model2.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs, 
                    callbacks=[model_checkpoint,reduce_lr, early_stopping],
                    verbose=2,      
                    use_multiprocessing=True,
                    workers=4)

#plot result
fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_score.plot(history.epoch, history.history["my_iou_metric_2"], label="Train score")
ax_score.plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation score")
ax_score.legend()
savefig(f'{TAG}_loss_score.png')

result = {'loss': history.history["loss"],
          'val_loss': history.history["val_loss"],
          'my_iou_metric_2': history.history["my_iou_metric_2"],
          'val_my_iou_metric_2': history.history["val_my_iou_metric_2"],
         }
df_result = pd.DataFrame.from_dict(result)
df_result.to_csv(f'{TAG}_result.csv')
