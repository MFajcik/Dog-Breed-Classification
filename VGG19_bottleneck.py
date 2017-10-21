import numpy as np
from keras.applications import VGG19
from keras.applications.vgg16 import preprocess_input, VGG16
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

import prologue
from tqdm import tqdm

#data_dir = r"C:\\Users\\mfajc\\Kaggle\\DogBreeds"
#grid, labels, train_idx, valid_idx, ytr, yv = prologue.init(data_dir)

INPUT_SIZE = 224
NUM_CLASSES= 120

data_dir = r"/home/ifajcik/kaggle/dog_breed_classification/dogbreed_data"
labels, train_idx, valid_idx, ytr, yv = prologue.init(data_dir)

x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
for i, img_id in tqdm(enumerate(labels['id'])):
    img = prologue.read_img(data_dir, img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x_train[i] = preprocess_input(img)
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))

vgg_bottleneck = VGG19(weights='imagenet', include_top=False, pooling='max')
train_vgg_bf = vgg_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_vgg_bf = vgg_bottleneck.predict(Xv, batch_size=32, verbose=1)

print('VGG train bottleneck features shape: {} size: {:,}'.format(train_vgg_bf.shape, train_vgg_bf.size))
print('VGG valid bottleneck features shape: {} size: {:,}'.format(valid_vgg_bf.shape, valid_vgg_bf.size))

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=prologue.SEED)
logreg.fit(train_vgg_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_vgg_bf)
valid_preds = logreg.predict(valid_vgg_bf)

print('Validation VGG LogLoss {}'.format(log_loss(yv, valid_probs)))
print('Validation VGG Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))