import numpy as np
from keras.applications import xception
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

import prologue
from tqdm import tqdm

# data_dir = r"C:\\Users\\mfajc\\Kaggle\\DogBreeds"
# grid, labels, train_idx, valid_idx, ytr, yv = prologue.init(data_dir)

INPUT_SIZE = 299
NUM_CLASSES = 120

x_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

data_dir = r"/home/ifajcik/kaggle/dog_breed_classification/dogbreed_data"
labels, y_train = prologue.init2(data_dir, NUM_CLASSES)

x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
for i, img_id in tqdm(enumerate(labels['id'])):
    img = prologue.read_img(data_dir, img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x_train[i] = xception.preprocess_input(img)
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

Xtr, Xv, Ytr, Yv = prologue.shuffle(x_train, y_train, 5)
print((Xtr.shape, Xv.shape, Ytr.shape, Yv.shape))

train_xception_bf = x_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_xception_bf = x_bottleneck.predict(Xv, batch_size=32, verbose=1)

print('Xception train bottleneck features shape: {} size: {:,}'.format(train_xception_bf.shape, train_xception_bf.size))
print('Xception valid bottleneck features shape: {} size: {:,}'.format(valid_xception_bf.shape, valid_xception_bf.size))

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=prologue.SEED)
logreg.fit(train_xception_bf, (Ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_xception_bf)
valid_preds = logreg.predict(valid_xception_bf)

print('Validation Xception LogLoss {}'.format(log_loss(Yv, valid_probs)))
print('Validation Xception Accuracy {}'.format(accuracy_score((Yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))
