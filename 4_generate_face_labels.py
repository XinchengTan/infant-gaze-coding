from fc_model import *
from data import *
import argparse
from PIL import Image

model_path = 'models/best_model/weights_last.pt'

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
model.eval()

video_files = list(video_folder.glob("*.mp4"))
for video_file in video_files:
    print(video_file.stem)
    files = list((dataset_folder / video_file.stem / 'img').glob(f'*.png'))
    img_filenames = [f.stem for f in files]
    idx = 0
    filenames = sorted(img_filenames)
    face_labels = np.load(str(Path.joinpath(dataset_folder, video_file.stem, 'face_labels.npy')))
    face_labels_fc = []
    for frame in range(face_labels.shape[0]):
        if face_labels[frame] < 0:
            face_labels_fc.append(face_labels[frame])
        else:
            faces = []
            while(int(img_filenames[idx][:5]) == frame):
                img = Image.open(dataset_folder / img_filenames[idx]).convert('RGB')
                img = data_transforms['val'](img)
                faces.append(img)
                idx += 1
            idx -= 1
            output = model(faces)
            pred = torch.max(output, 1)
            face_labels_fc.append(output)
    face_labels_fc = np.array(face_labels_fc)
    np.save(str(Path.joinpath(dataset_folder, video_file.stem, 'face_labels_fc.npy')), face_labels_fc)