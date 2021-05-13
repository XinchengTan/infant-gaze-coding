import numpy as np
from tqdm import tqdm
import shutil
from config import *

multi_face_folder.mkdir()
names = [f.stem for f in Path(video_folder).glob('*.mp4')]
face_hist = np.zeros(10)
total_datapoint = 0
for name in names:
    print(name)
    src_folder = dataset_folder / name
    dst_folder = multi_face_folder / name
    dst_folder.mkdir()
    (dst_folder / 'img').mkdir()
    (dst_folder / 'box').mkdir()
    face_labels = np.load(src_folder / 'face_labels.npy')
    files = list((src_folder / 'img').glob(f'*.png'))
    filenames = [f.stem for f in files]
    filenames = sorted(filenames)
    num_datapoint = 0
    for i in tqdm(range(len(filenames))):
        if filenames[i][-1] != '0':
            face_label = face_labels[int(filenames[i][:5])]
            num_datapoint += 1
            total_datapoint += 1
            faces = [filenames[i-1]]
            while i < len(filenames) and filenames[i][-1] != '0':
                faces.append(filenames[i])
                i += 1
            face_hist[len(faces)] += 1
            for face in faces:
                shutil.copy((src_folder / 'img' / (face + '.png')),
                            (dst_folder / 'img' / f'{name}_{face}_{face_label}.png'))
                shutil.copy((src_folder / 'box' / (face + '.npy')),
                            (dst_folder / 'box' / f'{name}_{face}_{face_label}.npy'))
    print('# multi-face datapoint:', num_datapoint)
print('total # multi-face datapoint:', total_datapoint)
print(face_hist)
