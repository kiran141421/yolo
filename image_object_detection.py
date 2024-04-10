from yolov8 import YOLOv8
import matplotlib.pyplot as plt
import numpy as np

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

onnx_file_path = "./models/yolov8m.onnx"
yolo_model = YOLOv8(onnx_file_path, conf_thres=0.5, iou_thres=0.5)

def yolo_object_detection(test_image):

    # test_image = cv.imread(test_image_path)

    objects = yolo_model(test_image)[2]

    names = []
    for i in objects:
        names.append(class_names[i])

    return names

    

    # detected_objects = yolo_model.draw_detections(test_image)
    # detected_objects = cv.cvtColor(detected_objects, cv.COLOR_BGR2RGB)

    # plt.subplot()
    # plt.imshow(detected_objects)
    # plt.axis("off")
    # plt.show()


test_image_paths = [
    "./images/baseball.jpg",
    "./images/people.jpg",
    "./images/giraffe-zebra.jpg",
    "./images/street.jpg",
]

test_image_path = test_image_paths[1]

if __name__ == "__main__":
    yolo_object_detection(test_image_path)
