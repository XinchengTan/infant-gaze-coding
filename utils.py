import os
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from config import *


def calculate_confusion_matrix(label, pred, save_path, class_num=3):
    mat = np.zeros([class_num, class_num])
    pred = np.array(pred)
    label = np.array(label)
    acc = sum(pred == label) / len(label)
    print('acc:{:.4f}'.format(acc))
    print('# datapoint', len(label))
    for i in range(class_num):
        for j in range(class_num):
            mat[i][j] = sum((label == i) & (pred == j))
    print(mat)
    mat = mat / np.sum(mat, -1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(mat, ax=ax, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    ax.set_xticklabels(['away', 'left', 'right'])
    ax.set_yticklabels(['away', 'left', 'right'])
    plt.axis('equal')
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)
    return acc


def confusion_mat(targets, preds, classes, normalize=False, plot=False, title="Confusion Matrix", cmap=plt.cm.Blues):
  cm = confusion_matrix(targets, preds)
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  if plot:
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + ".png")
    plt.show()

  return cm


def plot_learning_curve(train_accs, val_accs, save_dir):
  epochs = np.arange(1, len(train_accs) + 1)
  plt.plot(epochs, train_accs, label="Training Accuracy")
  plt.plot(epochs, val_accs, label="Validation Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.savefig(os.path.join(save_dir, 'learning_curve.png'))


def print_dataImg_name(dataloaders, dl_key, selected_idxs=None):
  dataset = dataloaders[dl_key].dataset
  fp_prefix = os.path.join(face_data_folder, dl_key)
  if selected_idxs is None:
    for fname, lbl in dataset.samples:
      print(fname.strip(fp_prefix))
  else:
    for i, tup in enumerate(dataset.samples):
      if i in selected_idxs:
        print(tup[0].strip(fp_prefix))

