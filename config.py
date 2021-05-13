from pathlib import Path

classes = {'away': 0, 'left': 1, 'right': 2}

face_model_file = Path("models", "face_model.caffemodel")
config_file = Path("models", "config.prototxt")

raw_folder = Path('/Users/Peng/Desktop/LookIt_raw')
database = Path('/Users/Peng/Desktop/LookIt')
video_folder = database / 'videos'
label_folder = database / 'coding_first'
label2_folder = database / 'coding_second'
dataset_folder = database / 'dataset'
multi_face_folder = database / 'multi_face'
