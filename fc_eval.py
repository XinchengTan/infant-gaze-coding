import os

import numpy as np
import torch
from tqdm import tqdm

from config import multi_face_folder
from utils import confusion_mat


def evaluate(args, model, dataloader, criterion, return_prob=False, is_labelled=False, generate_labels=True):
  model.eval()
  running_loss = 0
  running_top1_correct = 0
  pred_labels, pred_probs = [], []
  target_labels = []

  # Iterate over data.
  for inputs, labels in tqdm(dataloader):
    if generate_labels:
      target_labels.extend(list(labels.numpy()))  # -- 1d array

    inputs = inputs.to(args.device)
    labels = labels.to(args.device)

    # track history if only in train
    with torch.set_grad_enabled(False):
      # Get model outputs and calculate loss
      outputs = model(inputs)
      if is_labelled:
        loss = criterion(outputs, labels)
      _, preds = torch.max(outputs, 1)  # Make prediction

      if return_prob:
        pred_probs.append(outputs.cpu().detach().numpy())
      if generate_labels:
        nparr = preds.cpu().detach().numpy()  # -- 1d array
        pred_labels.extend(nparr)

    if is_labelled:
      running_loss += loss.item() * inputs.size(0)
      running_top1_correct += torch.sum(preds == labels.data)
    else:
      pass

  if is_labelled:
    epoch_loss = float(running_loss / len(dataloader.dataset))
    epoch_top1_acc = float(running_top1_correct.double() / len(dataloader.dataset))
  else:
    epoch_loss = None
    epoch_top1_acc = None

  if return_prob:
    print("Predicted label softmax output:", pred_probs)

  # Show confusion matrix
  if generate_labels and is_labelled:
    print("pred labels:", np.shape(pred_labels), pred_labels)
    print("target labels:", np.shape(target_labels), target_labels)
    cm = confusion_mat(target_labels, pred_labels, classes=['infant', 'others'])
    print("Confusion matrix:\n", cm)

  return epoch_loss, epoch_top1_acc, pred_labels, pred_probs, target_labels


# TODO: predict on a subset
def predict_on_minibatch(args, test_imgs, model, criterion):
  # apply test data transform

  # predict via model(loaded_test_img)  -- TODO: verify len(test_imgs) can < batch size
  pass


def predict_on_test(args, model, dataloaders, criterion):

  # Get predictions for the test set
  _, _, test_labels, test_probs, _ = evaluate(args, model, dataloaders['test'], criterion,
                                              return_prob=False, is_labelled=False, generate_labels=True)

  ''' These convert your dataset labels into nice human readable names '''

  def label_number_to_name(lbl_ix):
    return dataloaders['val'].dataset.classes[lbl_ix]

  # TODO: modify this
  def dataset_labels_to_names(dataset_labels, dataset_name):
    # dataset_name is one of 'train','test','val'
    dataset_root = os.path.join(multi_face_folder, dataset_name)
    found_files = []
    for parentdir, subdirs, subfns in os.walk(dataset_root):
      parentdir_nice = os.path.relpath(parentdir, dataset_root)
      found_files.extend([os.path.join(parentdir_nice, fn) for fn in subfns if fn.endswith('.png')])
    # Sort alphabetically, this is the order that our dataset will be in
    found_files.sort()
    # Now we have two parallel arrays, one with names, and the other with predictions
    assert len(found_files) == len(dataset_labels), "Found more files than we have labels"
    preds = {os.path.basename(found_files[i]): list(map(label_number_to_name, dataset_labels[i])) for i in
             range(len(found_files))}
    return preds

  output_test_labels = "test_set_predictions"
  output_salt_number = 0

  output_label_dir = "."

  while os.path.exists(os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number))):
    output_salt_number += 1
    # Find a filename that doesn't exist

  # test_labels_js = dataset_labels_to_names(test_labels, "test")
  # with open(os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number)), "w") as f:
  #   json.dump(test_labels_js, f, sort_keys=True, indent=4)

  print("Wrote predictions to:\n%s" % os.path.abspath(
    os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number))))

  return

