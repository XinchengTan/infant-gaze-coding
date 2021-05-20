# Automated Gaze Coding for Infant Videos
`0_filter_raw_data.py` organizes the raw videos downloaded from the Lookit platform. It puts the videos with annotations into `videos` folder and the annotation from the first and second human annotators into `coding_first` and `coding_second` folders respectively.

`1_create_datast.py` creates the dataset of faces extracted from the videos in the foler `dataset`. There will be a sub-folder for each video in `dataset`. In the folder for each video, a `img` folder stores all of the image patches extracted in the video with filenames `{frame_number}_{face_number}.png`. The corresponding spatial feature of bounding box for each image patch is stored in `box/{frame_number}_{face_number}.png`. `gaze_labels.npy` stores the gaze direction label for each frame labeled by the first annotator and `face_labels.npy` stores the face label generated by the lowest-face selection mechanism. If the video has annotation from the second annotator, there will be a `gaze_labels_second.npy` file to store it.

`2_generate_multi_face_subset.py` extracts all of image patches and the spatial features of their bounding boxes from the frames where more than one face is detected from the whole dataset. There will be a sub-folder for each video in the `multi_face` folder. Each image patch and box feature is stored with filename `img/{video_name}_{frame_number}_{face_number}_{face_label}.img` and `box/{video_name}_{frame_number}_{face_number}_{face_label}.npy` respectively, where the face label is generated by the lowest-face mechanism.

`3_train_face_classifier.py` trains infant classifiers with the data in `infant_vs_others` dataset, which is organized as the standard data folder for Pytorch Dataset.

`4_generate_face_labels.py` generates the face labels for each frame using the trained face classifer and the nearest patch mechanism. The labels will be stored in `dataset/{video_name}/face_labels_fc.npy`.

`5_train_gaze_classifier.py` trains the gaze classifier based on the data generated by the above steps.
