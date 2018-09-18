import numpy as np
import pandas as pd
from func import custom_loss
from func import img_process
from importlib import reload
from keras.models import Model, load_model, save_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def plot_history_result(history, TAG, name):
    #plot result
   
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history["my_iou_metric_2"], label="Train score")
    ax_score.plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation score")
    ax_score.legend()
    savefig(f'{TAG}_loss_score_{name}.png')
    matplotlib.pyplot.clf()

    result = {'loss': history.history["loss"],
              'val_loss': history.history["val_loss"],
              'my_iou_metric_2': history.history["my_iou_metric_2"],
              'val_my_iou_metric_2': history.history["val_my_iou_metric_2"],
             }
    df_result = pd.DataFrame.from_dict(result)
    df_result.to_csv(f'{TAG}_result_{name}.csv')
    # fine tune threshold

def fine_tune_threshold(save_model_name, val_df, TAG):
    print('fine tune threshold')
    model = load_model(save_model_name,custom_objects={'my_iou_metric_2': custom_loss.my_iou_metric_2,
                                                       'lovasz_loss': custom_loss.lovasz_loss})
    x_valid = img_process.read_train_img_to_np_array(val_df)
    y_valid = img_process.read_mask_img_to_np_array(val_df)

    preds_valid = predict_result_reflect_TTA(model, x_valid, 128)

    y_valid_pad = y_valid[:,14:-13, 14:-13,:]
    preds_valid_pad = preds_valid[:,14:-13, 14:-13]

    ## Scoring for last model, choose threshold by validation data 
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

    # ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
    # print(ious)
    ious = np.array([custom_loss.iou_metric_batch(y_valid_pad, preds_valid_pad > threshold) for threshold in thresholds])
    print(ious)

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
    savefig(f'{TAG}_threshold.png')
    matplotlib.pyplot.clf()
    
    thresholds = {'thresholds': thresholds, 'ious': ious}
    df_thresholds = pd.DataFrame.from_dict(thresholds)
    df_thresholds.to_csv(f'{TAG}_thresholds.csv')

def predict_result_reflect_TTA(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2