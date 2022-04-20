import os
import json
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2

np.random.seed(41)

classname_to_id = {"person": 1}
labels = ["LANK",  # 0 left ankle - upper
          "LWRA",  # 1 left wrist marker a - upper
          "LUPA",  # 2 left upper arm - upper
          "RTIB",  # 3 right tibia - lower
          "CLAV",  # 4 clavicle - upper
          "LTHI",  # 5 left thigh - lower
          "RHEE",  # 6 right heel - lower
          "RELB",  # 7 right elbow - upper
          "LFIN",  # 8 left finger - upper
          "LKNE",  # 9 left knee - lower
          "RWRA",  # 10 right wrist marker a - upper
          "RWRB",  # 11 right wrist marker b - upper
          "C7",  # 12 7th cervical vertebra - upper
          "LSHO",  # 13 left shoulder - upper
          "RFIN",  # 14 right finger - upper
          "RPSI",  # 15 right posterior superior iliac - lower
          "LTIB",  # 16 left tibia - lower
          "RASI",  # 17 right anterior superior iliac - lower
          "LELB",  # 18 left elbow - upper
          "RTOE",  # 19 right toe - lower
          "RBAK",  # 20 right back - upper
          "T10",  # 21 10th thoracic vertebra - upper
          "RFRM",  # 22 right forearm - upper
          "LHEE",  # 23 left heel - lower
          "LPSI",  # 24 left posterior superior iliac - lower
          "LASI",  # 25 left anterior superior iliac - lower
          "RKNE",  # 26 right knee - lower
          "RUPA",  # 27 right upper arm - upper
          "RANK",  # 28 right ankle - lower
          "LTOE",  # 29 left toe - lower
          "RTHI",  # 30 right thigh - lower
          "STRN",  # 31 sternum - upper
          "LWRB",  # 32 left wrist marker b - upper
          "LFRM",  # 33 left forearm - upper
          "RSHO"]  # 34 right shoulder - upper


class Lableme2CoCo:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def to_coco(self, json_path_list):
        self._init_categories()
        instance = {}
        instance['info'] = {'description': 'AIK Pose Estimation Dataset',
                            'version': 1.0,
                            'year': 2022,
                            'contributor': "yuan zi",
                            'date_created': "2022/04/17"}
        instance['license'] = ['yuan zi']
        instance['images'] = self.images
        instance['categories'] = self.categories

        for json_path in (json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']

            # check number of people in the image
            num_person = 0
            isIndividual = False
            for shape in shapes:
                if shape['group_id'] == None:
                    isIndividual = True
                    continue
                if shape['group_id'] > num_person:
                    num_person = shape['group_id']

            print("Start annotate for img: ", json_path, "There are", num_person + 1, "people in total")
            # do annotation for each person
            for person in range(num_person + 1):
                print("Person", person + 1, "...")
                # start with person = 0, create annotation dict for each person
                person_annotation = []
                keypoints = [None] * 35

                part_index = 0
                for shape in shapes:
                    # iterate through keypoints, add to dict if belongs to person
                    # if shape['group_id'] != person and isIndividual == False:
                    #     continue
                    # get the body part this keypoint represents

                    # store the keypoint data to keypoints[] at its respective index
                    keypoints[part_index] = shape['points'][0]
                    part_index = part_index + 1

                # edit the keypoint data to fit COCO annotation format
                num_keypoints = 0
                for keypoint_i in range(35):
                    # store keypoint for person in annotation
                    if keypoints[keypoint_i] == None:
                        person_annotation.extend([0, 0, 0])
                    else:
                        person_annotation.extend([keypoints[keypoint_i][0], keypoints[keypoint_i][1], 2])
                        num_keypoints += 1

                        # annotate all other information for this person
                annotation = {}
                annotation['id'] = self.ann_id
                annotation['image_id'] = self.img_id
                annotation['category_id'] = 1
                annotation['iscrowd'] = 0
                annotation['num_keypoints'] = num_keypoints
                annotation['keypoints'] = person_annotation

                # detect bbx by using mediapipe and use it as label
                mp_pose = mp.solutions.pose
                # For static images:
                with mp_pose.Pose(static_image_mode=True,
                                  model_complexity=1,
                                  enable_segmentation=True,
                                  min_detection_confidence=0.5) as pose:

                    image = cv2.imread(json_path.split('.json')[0] + '.' + obj['imagePath'].split('.')[-1])

                    image_height, image_width, _ = image.shape
                    # Convert the BGR image to RGB before processing.
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # mask_img = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    mask_img = np.array(results.segmentation_mask, dtype=np.uint8) * 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))  # 定义矩形结构元素

                    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算1

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 定义矩形结构元素
                    mask_img = cv2.dilate(mask_img, kernel, iterations=3)
                    thresh = cv2.Canny(mask_img, 128, 256)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
                    x, y, w, h = cv2.boundingRect(contours[0])

                annotation['bbox'] = [x, y, w, h ]

                # add person annotation to image annotation
                # print("Annotated data: ", annotation)
                annotation['area'] = w*h
                self.annotations.append(annotation)
                self.ann_id += 1

            # next image
            self.img_id += 1

        # store to output .json instance
        instance['annotations'] = self.annotations
        return instance

    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['supercategory'] = k
            category['id'] = v
            category['name'] = k
            category['keypoints'] = ["LANK",  # 0 left ankle - upper
                                     "LWRA",  # 1 left wrist marker a - upper
                                     "LUPA",  # 2 left upper arm - upper
                                     "RTIB",  # 3 right tibia - lower
                                     "CLAV",  # 4 clavicle - upper
                                     "LTHI",  # 5 left thigh - lower
                                     "RHEE",  # 6 right heel - lower
                                     "RELB",  # 7 right elbow - upper
                                     "LFIN",  # 8 left finger - upper
                                     "LKNE",  # 9 left knee - lower
                                     "RWRA",  # 10 right wrist marker a - upper
                                     "RWRB",  # 11 right wrist marker b - upper
                                     "C7",  # 12 7th cervical vertebra - upper
                                     "LSHO",  # 13 left shoulder - upper
                                     "RFIN",  # 14 right finger - upper
                                     "RPSI",  # 15 right posterior superior iliac - lower
                                     "LTIB",  # 16 left tibia - lower
                                     "RASI",  # 17 right anterior superior iliac - lower
                                     "LELB",  # 18 left elbow - upper
                                     "RTOE",  # 19 right toe - lower
                                     "RBAK",  # 20 right back - upper
                                     "T10",  # 21 10th thoracic vertebra - upper
                                     "RFRM",  # 22 right forearm - upper
                                     "LHEE",  # 23 left heel - lower
                                     "LPSI",  # 24 left posterior superior iliac - lower
                                     "LASI",  # 25 left anterior superior iliac - lower
                                     "RKNE",  # 26 right knee - lower
                                     "RUPA",  # 27 right upper arm - upper
                                     "RANK",  # 28 right ankle - lower
                                     "LTOE",  # 29 left toe - lower
                                     "RTHI",  # 30 right thigh - lower
                                     "STRN",  # 31 sternum - upper
                                     "LWRB",  # 32 left wrist marker b - upper
                                     "LFRM",  # 33 left forearm - upper
                                     "RSHO"]  # 34 right shoulder - upper

            # category['skeleton'] = [
            #     [16, 14],
            #     [14, 12],
            #     [17, 15],
            #     [15, 13],
            #     [12, 13],
            #     [6, 12],
            #     [7, 13],
            #     [6, 7],
            #     [6, 8],
            #     [7, 9],
            #     [8, 10],
            #     [9, 11],
            #     [2, 3],
            #     [1, 2],
            #     [1, 3],
            #     [2, 4],
            #     [3, 5],
            #     [4, 6],
            #     [5, 7]
            # ]
            self.categories.append(category)

    def _image(self, obj, path):
        image = {}
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # image["height"] = img_x.shape[0]
        # image["width"] = img_x.shape[1]
        image["height"] = obj['imageHeight']
        image["width"] = obj['imageWidth']
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # read json file, return json object
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)


if __name__ == '__main__':
    print("---------------------------------")
    train = Lableme2CoCo()
    val = Lableme2CoCo()

    # name of the folders containing labelme format json files
    folders = [
        "/home/yzi/PycharmProjects/human_pose/data/annotation_dataset/AIK_dataset_aik10_normal/content/AIK_dataset_labelme"]
    save_folder = '/home/yzi/PycharmProjects/human_pose/data/annotation_dataset/AIK_dataset_aik10_normal/content/AIK_dataset_coco/annotation'

    # loop through the directories and start converting
    for folder in folders:
        print("Saving in ", folder)

        # set the paths
        json_path = os.path.join(folder)  # path to labelme json folder
        json_list_path = glob.glob(json_path + "/*.json")  # labelme json files in folder
        train_path, val_path = train_test_split(json_list_path, test_size=0.2)  # split to train and test data set
        train_save_path = save_folder + "/train_AIK_10_normal.json"  # path to save COCO json files (train)
        val_save_path = save_folder + "/val_AIK_10_normal.json"  # path to save COCO json files (validation)

        # convert to COCO format
        train_instance = train.to_coco(train_path)
        val_instance = val.to_coco(val_path)

        # save the converted COCO json files
        json.dump(train_instance, open(train_save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        json.dump(val_instance, open(val_save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
