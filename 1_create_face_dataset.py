import numpy as np
import cv2
import time
import logging
from joblib import Parallel, delayed
from config import *


def parse_label(file, fps):
    labels = np.genfromtxt(open(file, "rb"), dtype='str', delimiter=",", skip_header=3)
    output = []
    for entry in range(labels.shape[0]):
        frame = int(int(labels[entry, 0]) * fps / 1000)
        class_name = labels[entry, 2]
        valid_flag = 1 if class_name in classes else 0
        frame_label = [frame, valid_flag, class_name]
        output.append(frame_label)
    output.sort(key=lambda x: x[0])
    if len(output):
        return output
    else:
        return None


def detect_face_opencv_dnn(net, frame, conf_threshold):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)
            y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)
            x2 = max(int(detections[0, 0, i, 5] * frameWidth), 0)
            y2 = max(int(detections[0, 0, i, 6] * frameHeight), 0)
            bboxes.append([x1, y1, x2-x1, y2-y1])
    return bboxes


def process_video(video_list):
    net = cv2.dnn.readNetFromCaffe(str(config_file), str(face_model_file))
    for video_file in video_list:
        st_time = time.time()
        logging.critical(f"Proccessing: {str(video_file)}")
        Path.joinpath(dataset_folder / video_file.stem).mkdir()
        img_folder = Path.joinpath(dataset_folder, video_file.stem, 'img')
        img_folder.mkdir()
        box_folder = Path.joinpath(dataset_folder, video_file.stem, 'box')
        box_folder.mkdir()

        frame_counter = 0
        no_face_counter = 0
        no_annotation_counter = 0
        valid_counter = 0
        gaze_labels = []
        face_labels = []

        cap = cv2.VideoCapture(str(video_file))
        responses = parse_label(label_folder / (video_file.stem + '.txt'), cap.get(cv2.CAP_PROP_FPS))
        ret_val, frame = cap.read()

        while ret_val:
            if responses:
                if frame_counter >= responses[0][0]:  # skip until reaching first annotated frame
                    # find closest (previous) response this frame belongs to
                    q = [index for index, val in enumerate(responses) if frame_counter >= val[0]]
                    response_index = max(q)
                    if responses[response_index][1] != 0:  # make sure response is valid
                        bbox = detect_face_opencv_dnn(net, frame, 0.7)
                        if not bbox:
                            no_face_counter += 1
                            gaze_labels.append(-1)
                            face_labels.append(-1)
                            logging.info("Face not detected in frame: " + str(frame_counter))
                        else:
                            # select lowest face, probably belongs to kid: face = min(bbox, key=lambda x: x[3] - x[1])
                            selected_face = 0
                            min_value = bbox[0][3] - bbox[0][1]
                            gaze_class = responses[response_index][2]
                            for i, face in enumerate(bbox):
                                if bbox[i][3] - bbox[i][1] < min_value:
                                    min_value = bbox[i][3] - bbox[i][1]
                                    selected_face = i
                                crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
                                resized_img = cv2.resize(crop_img, (100, 100))
                                face_box = np.array([face[1], face[1] + face[3], face[0], face[0] + face[2]])
                                img_shape = np.array(frame.shape)
                                ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                                                  face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
                                face_size = (ratio[1] - ratio[0]) * (ratio[3] - ratio[2])
                                face_ver = (ratio[0] + ratio[1]) / 2
                                face_hor = (ratio[2] + ratio[3]) / 2
                                face_height = ratio[1] - ratio[0]
                                face_width = ratio[3] - ratio[2]
                                feature_dict = {
                                    'face_box': face_box,
                                    'img_shape': img_shape,
                                    'face_size': face_size,
                                    'face_ver': face_ver,
                                    'face_hor': face_hor,
                                    'face_height': face_height,
                                    'face_width': face_width
                                }
                                cv2.imwrite(str(img_folder / f'{frame_counter:05d}_{i:01d}.png'), resized_img)
                                np.save(str(box_folder / f'{frame_counter:05d}_{i:01d}.npy'), feature_dict)
                            valid_counter += 1
                            gaze_labels.append(classes[gaze_class])
                            face_labels.append(selected_face)
                            logging.info(f"valid frame in class {gaze_class}")

                    else:
                        no_annotation_counter += 1
                        gaze_labels.append(-2)
                        face_labels.append(-2)
                        logging.info("Skipping since frame is invalid")
                else:
                    no_annotation_counter += 1
                    gaze_labels.append(-2)
                    face_labels.append(-2)
                    logging.info("Skipping since no annotation (yet)")
            else:
                gaze_labels.append(-2)
                face_labels.append(-2)
                no_annotation_counter += 1
                logging.info("Skipping frame since parser reported no annotation")
            ret_val, frame = cap.read()
            frame_counter += 1
            logging.info("Processing frame: {}".format(frame_counter))
        gaze_labels = np.array(gaze_labels)
        face_labels = np.array(face_labels)
        np.save(str(Path.joinpath(dataset_folder, video_file.stem, 'gaze_labels.npy')), gaze_labels)
        np.save(str(Path.joinpath(dataset_folder, video_file.stem, 'face_labels.npy')), face_labels)
        logging.critical("Total frame: {}, No face: {}, No annotation: {}, Valid: {}".format(frame_counter, no_face_counter, no_annotation_counter, valid_counter))
        ed_time = time.time()
        logging.critical('Time used: {}'.format(ed_time - st_time))


video_files = list(video_folder.glob("*.mp4"))
logging.critical(f"{len(video_files)} videos in total.")
dataset_folder.mkdir()
process_video(video_files)
