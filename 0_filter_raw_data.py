import shutil
from config import *

database.mkdir()
video_folder.mkdir()
label_folder.mkdir()
label2_folder.mkdir()

prefix = 'cfddb63f-12e9-4e62-abd1-47534d6c4dd2_'
coding_first = [f.stem[:-5] for f in Path(raw_folder / 'coding_first').glob('*.txt')]
coding_second = [f.stem[:-5] for f in Path(raw_folder / 'coding_second').glob('*.txt')]
videos = [f.stem for f in Path(raw_folder / 'videos').glob(prefix+'*.mp4')]

print('coding_fist:', len(coding_first))
print('coding_second:', len(coding_second))
print('videos: ', len(videos))

training_set = []
test_set = []

for filename in videos:
    if prefix not in filename:
        continue
    label_id = filename[len(prefix):]
    if label_id in coding_first:
        if label_id in coding_second:
            test_set.append(filename)
        else:
            training_set.append(filename)

print('training set:', len(training_set), 'validation set:', len(test_set))

for filename in training_set:
    shutil.copyfile(raw_folder / 'videos' / (filename+'.mp4'), database / 'videos'/ (filename[len(prefix):]+'.mp4'))
    shutil.copyfile(raw_folder / 'coding_first' / (filename[len(prefix):]+'-evts.txt'), database / 'coding_first'/ (filename[len(prefix):]+'.txt'))

for filename in test_set:
    shutil.copyfile(raw_folder / 'videos' / (filename + '.mp4'), database / 'videos'/(filename[len(prefix):]+'.mp4'))
    shutil.copyfile(raw_folder / 'coding_first' / (filename[len(prefix):] + '-evts.txt'), database / 'coding_first'/(filename[len(prefix):]+'.txt'))
    shutil.copyfile(raw_folder / 'coding_second' / (filename[len(prefix):] + '-evts.txt'), database / 'coding_second'/(filename[len(prefix):]+'.txt'))
