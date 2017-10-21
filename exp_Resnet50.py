import numpy as np
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions
import prologue
from mpl_toolkits.axes_grid1 import ImageGrid

data_dir = r"C:\\Users\\mfajc\\Kaggle\\DogBreeds"
#data_dir = r"/home/ifajcik/kaggle/dog_breed_classification/dogbreed_data"
labels= prologue.init(data_dir)

# plot image figure via ImageGrid
fig = plt.figure(1, figsize=(16, 16))
j = int(np.sqrt(prologue.NUM_CLASSES))
i = int(np.ceil(prologue.NUM_CLASSES / j))
grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)

#Pretrained resnet
model = ResNet50(weights='imagenet')

for i, (img_id, breed) in enumerate(labels.loc[labels['rank'] == 1, ['id', 'breed']].values):
    ax = grid[i]
    img = prologue.read_img(data_dir,img_id, 'train', (224, 224))
    x = preprocess_input(img.copy())
    ax.imshow(img / 255.)
    x = np.expand_dims(img, axis=0)
    preds = model.predict(x)
    _, imagenet_class_name, prob = decode_predictions(preds, top=1)[0][0]
    ax.text(10, 180, 'ResNet50: %s (%.2f)' % (imagenet_class_name , prob), color='w', backgroundcolor='k', alpha=0.8)
    ax.text(10, 200, 'LABEL: %s' % breed, color='k', backgroundcolor='w', alpha=0.8)
    ax.axis('off')
plt.show()