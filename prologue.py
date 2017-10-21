import numpy as np
import pandas as pd
from keras.preprocessing import image
from os.path import join
import matplotlib.pyplot as plt


def read_img(data_dir, img_id, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img


SEED = 1987
def init(data_dir,num_classes = None):
    labels = pd.read_csv(join(data_dir, 'labels.csv'))
    if num_classes is None:
        selected_breed_list = list(
            labels.groupby('breed').count().sort_values(by='id', ascending=False).index)
    else:
        selected_breed_list = list(
            labels.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index)
    labels = labels[labels['breed'].isin(selected_breed_list)]
    # Some wierd way to create one hot vectors
    labels['target'] = 1
    labels['rank'] = labels.groupby('breed').rank()['id']
    labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
    np.random.seed(seed=SEED)
    rnd = np.random.random(len(labels))
    train_idx = rnd < 0.8
    valid_idx = rnd >= 0.8
    y_train = labels_pivot[selected_breed_list].values
    ytr = y_train[train_idx]
    yv = y_train[valid_idx]
    return labels,train_idx,valid_idx, ytr, yv