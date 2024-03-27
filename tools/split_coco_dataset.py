import os
import shutil
import json
import numpy as np

path_to_coco_full_dataset_src = "/media/abhishek/data/college/thesis/thesis_with_dr_duc/datasets/COCODIR/coco/images/train2017"
train_annotations_file = "../data/coco/annotations/person_keypoints_train2017.json"
path_to_coco_subset_dest = "/media/abhishek/data/college/thesis/thesis_with_dr_duc/datasets/COCODIR/coco/images/train2017_10pc_set1"
train_annotations_subset_file = "person_keypoints_train2017_subset_10pc_set1.json"

#percentage of original dataset to keep in subset
subset_perc = 10

# list all image filenames in source (full) dataset
train_img_list = os.listdir(path_to_coco_full_dataset_src)

# create directory for subset
if not os.path.exists(path_to_coco_subset_dest):
    os.mkdir(path_to_coco_subset_dest)

total_num_images = len(train_img_list)
subset_num_images = (int) (subset_perc/100) * total_num_images

# TODO make this a random selection - than selecting first n images
# subset_3000_img_filenames = train_img_list[:3000]
subset_img_filenames = np.random.choice(train_img_list, subset_num_images, replace=False)
for img in subset_img_filenames:
    src_img_path = os.path.join(path_to_coco_full_dataset_src, img)
    dest_img_path = os.path.join(path_to_coco_subset_dest, img)
    shutil.copy(src_img_path, dest_img_path)

with open(train_annotations_file, "r+") as ann_fd:
    file_contents = ann_fd.read()
    train_annotations = json.loads(file_contents)

selected_ims   = []
selected_img_ids = []
for im in train_annotations['images']:
    if im['file_name'] in subset_img_filenames:
        selected_ims.append(im)
        selected_img_ids.append(im['id'])
selected_anns = []
for ann in train_annotations['annotations']:
    if ann['image_id'] in selected_img_ids:
        selected_anns.append(ann)

train_annotations_subset = {}
for key in train_annotations.keys():
    if key not in ["images", "annotations"]:
        train_annotations_subset[key] = train_annotations[key]
train_annotations_subset['images'] = selected_ims
train_annotations_subset['annotations'] = selected_anns

with open(train_annotations_subset_file, "w+") as subset_json_file:
    subset_json_file.write(json.dumps(train_annotations_subset))