import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.dataloader as dataloader
import argparse
from tqdm import tqdm
from config import *
from data import LookItDataset
from model import GazeCodingModel, GazeCodingModel3D
import utils


def check_all_same(seg):
    for i in range(1, seg.shape[0]):
        if seg[i] != seg[i-1]:
            return False
    return True


def generate_train_test_list(n_frames, step, face_label_name, eliminate_transition):
    all_names = [f.stem for f in label_folder.glob('*.txt')]
    test_names = [f.stem for f in label2_folder.glob('*.txt')]
    train_list = []
    val_list = []
    for name in tqdm(all_names):
        gaze_labels = np.load(str(Path.joinpath(dataset_folder, name, f'gaze_labels.npy')))
        face_labels = np.load(str(Path.joinpath(dataset_folder, name, f'{face_label_name}.npy')))
        for frame_number in range(gaze_labels.shape[0]):
            gaze_label_seg = gaze_labels[frame_number:frame_number + n_frames]
            face_label_seg = face_labels[frame_number:frame_number + n_frames]
            if len(gaze_label_seg) != n_frames:
                break
            if sum(face_label_seg < 0):
                continue
            if not eliminate_transition or check_all_same(gaze_label_seg):
                class_seg = gaze_label_seg[n_frames // 2]
                img_files_seg = []
                box_files_seg = []
                for i in range(n_frames):
                    img_files_seg.append(f'{name}/img/{frame_number+i:05d}_{face_label_seg[i]:01d}.png')
                    box_files_seg.append(f'{name}/box/{frame_number+i:05d}_{face_label_seg[i]:01d}.npy')
                img_files_seg = img_files_seg[::step]
                box_files_seg = box_files_seg[::step]
                if name in test_names:
                    val_list.append((img_files_seg, box_files_seg, class_seg))
                else:
                    train_list.append((img_files_seg, box_files_seg, class_seg))
    return train_list, val_list


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-m', type=str)
parser.add_argument('--device', '-d', type=str, default='cpu')
parser.add_argument('--face', '-f', type=str, default='face_labels')
parser.add_argument('--n_frames', '-n', type=int, default=10)
parser.add_argument('--step', '-s', type=int, default=2)
parser.add_argument('--add_box', default=False, action='store_true')
parser.add_argument('--del_tran', default=False, action='store_true')
parser.add_argument('--arch', type=str, default='2d')
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

train_list, val_list = generate_train_test_list(args.n_frames, args.step, args.face, args.del_tran)
save_folder = Path('models', args.model_name)
save_folder.mkdir()
print(f'# Datapoint: Train {len(train_list)} Val {len(val_list)}')
with open((save_folder / 'log.txt'), 'a+') as f:
    f.write(f'# Datapoint: Train {len(train_list)} Val {len(val_list)}\n')

batch_size = 128
datasets = {'train': LookItDataset(data_list=train_list, is_train=True),
            'valid': LookItDataset(data_list=val_list, is_train=False)}
dataloaders = {phase: dataloader.DataLoader(datasets[phase], batch_size=batch_size, shuffle=True, num_workers=16)
               for phase in ['train', 'valid']}

if args.arch == '3d':
    model = GazeCodingModel3D(device=args.device, n=args.n_frames//args.step, add_box=args.add_box).to(args.device)
else:
    model = GazeCodingModel(device=args.device, n=args.n_frames//args.step, add_box=args.add_box).to(args.device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 15], gamma=0.1)
criterion = nn.CrossEntropyLoss()

print('Start Training ...')
best_model_wts = copy.deepcopy(model.state_dict())
last_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(20):
    print(f'Epoch {epoch}:')
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0
        num_datapoints = 0

        for data in tqdm(dataloaders[phase]):
            labels = data['label'].to(args.device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(data)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * labels.size(0)
            num_datapoints += labels.size(0)
            running_corrects += torch.sum(torch.eq(preds, labels)).item()

        epoch_loss = running_loss / num_datapoints
        epoch_acc = running_corrects / num_datapoints
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        with open((save_folder / 'log.txt'), 'a+') as f:
            f.write(f'Epoch{epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

        if phase == 'train':
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            print('Learning rate:', lr)

        if phase == 'valid':
            last_model_wts = copy.deepcopy(model.state_dict())
            torch.save(last_model_wts, (save_folder / 'weights_last.pt'))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, (save_folder / 'weights_best.pt'))

print('Done!')

print('Evaluating last model...')
model.load_state_dict(torch.load((save_folder / 'weights_last.pt')))
model.eval()
gaze_label = []
gaze_pred = []
for data in tqdm(dataloaders['valid']):
    labels = data['label']
    gaze_label.extend(list(labels))
    with torch.set_grad_enabled(False):
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
    gaze_pred.extend(list(preds.cpu()))
last_acc = utils.calculate_confusion_matrix(gaze_label, gaze_pred, (save_folder / 'last_conf.pdf'))

print('Evaluating best model...')
model.load_state_dict(torch.load((save_folder / 'weights_best.pt')))
model.eval()
gaze_label = []
gaze_pred = []
for data in tqdm(dataloaders['valid']):
    labels = data['label']
    gaze_label.extend(list(labels))
    with torch.set_grad_enabled(False):
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
    gaze_pred.extend(list(preds.cpu()))
best_acc = utils.calculate_confusion_matrix(gaze_label, gaze_pred, (save_folder / 'best_conf.pdf'))

print(f'Best Val Acc: {best_acc:.4f}')
print(f'Last Val Acc: {last_acc:.4f}')
with open((save_folder / 'log.txt'), 'a+') as f:
    f.write(f'Best Val Acc: {best_acc:.4f}\n')
    f.write(f'Last Val Acc: {last_acc:.4f}\n')
