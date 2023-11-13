import cv2
from imread_from_url import imread_from_url
import signal
import sys

def signal_int_handle(signum,frame):
    print('Signal handler called with signal', signum)
    cv2.destroyAllWindows()
    sys.exit(0)
from yoloseg import YOLOSeg

# Initialize YOLOv5 Instance Segmentator
model_path = "models/yolov8s-seg.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)
signal.signal(signal.SIGINT, signal_int_handle)
# Read image
img_url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Giraffes_at_west_midlands_safari_park.jpg"
# img_url = "https://img95.699pic.com/photo/50144/8136.jpg_wh860.jpg"
img = imread_from_url(img_url)
# img = cv2.imread('2.jpg')
# Detect Objects
boxes, scores, class_ids, masks = yoloseg(img)

# Draw detections
combined_img = yoloseg.draw_masks(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)

while True:
    key = cv2.waitKey(100)
    # print(key)
    if key != -1:
        print(key)
        break
    # time.sleep(0.1)

cv2.destroyAllWindows()
