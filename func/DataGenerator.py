import keras
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from func.img_process import _img_augmentation, _convert_to_np_array

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=32, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
#         print('lenght')
#         print(int(np.floor((self.df.shape[0]) / self.batch_size)))
        return int(np.floor((self.df.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        'index: indicate which batch in a epoch, (first, second or third batch)'
        # Generate indexes of the batch
        batch_df = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size].copy()
        batch_df["img"] = [np.array(load_img("../data/train/{}.png".format(idx), grayscale=True)) / 255 for idx in batch_df.id]
        batch_df["img_mask"] = [np.array(load_img("../data/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in batch_df.id]
        
        augment_df = _img_augmentation(batch_df)
        X_np, y_np = _convert_to_np_array(augment_df)

        return X_np, y_np

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print(f', epoch={self.epoch}')
        self.epoch += 1