from fc_model import *
from fc_eval import *
from utils import *
from data import *
import argparse
from PIL import Image
from tqdm import tqdm

model_path = 'models/weights-Arohe-Mvgg16-D0-Osgd-Snone-L0.001-B8/weights_last.pt'

val_infant_files = [f.stem for f in (face_data_folder / 'val' / 'infant').glob('*.png')]
val_others_files = [f.stem for f in (face_data_folder / 'val' / 'others').glob('*.png')]
num_correct = 0
total = len(val_infant_files)+len(val_others_files)
for f in val_infant_files:
  if f[-1] == f[-3]:
    num_correct += 1
for f in val_others_files:
  if f[-1] != f[-3]:
    num_correct += 1
print(num_correct, total, num_correct / total)

parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--device', '-d', default='cuda:0', type=str)
# augmentations
parser.add_argument('--rotation', default=False, action='store_true')
parser.add_argument('--cropping', default=False, action='store_true')
parser.add_argument('--hor_flip', default=False, action='store_true')
parser.add_argument('--ver_flip', default=False, action='store_true')
parser.add_argument('--color', default=False, action='store_true')
parser.add_argument('--erasing', default=False, action='store_true')
parser.add_argument('--noise', default=False, action='store_true')
# model architecture
parser.add_argument('--model', default='vgg16', type=str)  # resnet, alexnet, vgg, squeezenet
# dropout
parser.add_argument('--dropout', default=0, type=float)  # can only be applied to resnet & densenet
args = parser.parse_args()

model, input_size = init_face_classifier(args, model_name=args.model, num_classes=2, resume_from=model_path)
data_transforms = get_fc_data_transforms(args, input_size)
dataloaders = get_dataset_dataloaders(args, input_size, 64, False)
criterion = get_loss()
model.to(args.device)

val_loss, val_top1, val_labels, val_probs, val_target_labels = evaluate(args, model, dataloaders['val'], criterion,
                                                                        return_prob=False, is_labelled=True, generate_labels=True)


print("\n[val] Failed images:")
err_idxs = np.where(np.array(val_labels) != np.array(val_target_labels))[0]
print_dataImg_name(dataloaders, 'val', err_idxs)
print(f'val_loss: {val_loss:.4f}', f'val_top1: {val_top1:.4f}')

video_files = list(video_folder.glob("*.mp4"))
for video_file in video_files:
    print(video_file.stem)
    files = list((dataset_folder / video_file.stem / 'img').glob(f'*.png'))
    filenames = [f.stem for f in files]
    filenames = sorted(filenames)
    idx = 0
    face_labels = np.load(str(Path.joinpath(dataset_folder, video_file.stem, 'face_labels.npy')))
    face_labels_fc = []
    hor, ver = 0.5, 1
    for frame in tqdm(range(face_labels.shape[0])):
        if face_labels[frame] < 0:
            face_labels_fc.append(face_labels[frame])
        else:
            faces = []
            centers = []
            while idx < len(filenames) and (int(filenames[idx][:5]) == frame):
                img = Image.open(dataset_folder / video_file.stem / 'img' / (filenames[idx]+'.png')).convert('RGB')
                box = np.load(dataset_folder / video_file.stem / 'box' / (filenames[idx]+'.npy'), allow_pickle=True).item()
                centers.append([box['face_hor'], box['face_ver']])
                img = data_transforms['val'](img)
                faces.append(img)
                idx += 1
            centers = np.stack(centers)
            faces = torch.stack(faces).to(args.device)
            model.eval()
            output = model(faces)
            _, preds = torch.max(output, 1)
            preds = preds.cpu().numpy()
            idxs = np.where(preds==0)[0]
            centers = centers[idxs]
            if centers.shape[0] == 0:
                face_labels_fc.append(-1)
            else:
                dis = np.sqrt((centers[:, 0] - hor) ** 2 + (centers[:, 1] - ver) ** 2)
                i = np.argmin(dis)
                face_labels_fc.append(idxs[i])
                hor, ver = centers[i]
    face_labels_fc = np.array(face_labels_fc)
    np.save(str(Path.joinpath(dataset_folder, video_file.stem, 'face_labels_fc.npy')), face_labels_fc)