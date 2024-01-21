import cv2
import numpy as np
from cvlib.object_detection import ObjectDetection
from cvlib.tracker import EuclideanDistTracker

video_name = "camera7"
video_capture = cv2.VideoCapture(f"./Videos/{video_name}.mp4")
# Name for class
class_ref = ["car", "motorbike"]
# Class to take set None to take all class
take_class = [0]
# Is render the video
render = True
# Video resize
width = 848
height = 477
dim = (width, height)


def image_resize(img, dim):
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


car_cascade = cv2.CascadeClassifier("./cascade/cars.xml")

fps = video_capture.get(cv2.CAP_PROP_FPS)

# Distance between two point to be consider as same object
base_dis = 25
# Skip frame-th to reduce the number of frame to process
targer_counter = 1
counter = 0
# Cut the frame to reduce the number of frame to process
frameCut = False
# cut the frame to this size to reduce time to process
# cap_x, cap_y, cap_w, cap_h = 200, 270, 420, 200
cap_x, cap_y, cap_w, cap_h = 0, 0, width, height
line = int(cap_h * 0.70)
show_cap = True
# Distance scalar to increase the distance between two point to be consider as same object in case of high speed and low fps
dis_scalar = (targer_counter + 1) / targer_counter if frameCut else 1
# Acceptable distance from line
acceptable = 10
# Video output in case of render is True
video_results = cv2.VideoWriter(
    f"Videos_ouput/{video_name}_result.avi",
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    fps,
    (width, height),
)


def draw_with_alpha(img, draw_function, up, lp, color, width, alpha=0.5):
    overlay = np.zeros_like(img, dtype=np.uint8)
    draw_function(overlay, up, lp, color, width)
    mask = overlay.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)[mask]


detector = ObjectDetection()

n_of_frame = 0
import time

tic = time.time()

while True:
    if frameCut:
        if counter == targer_counter:
            counter = 0
            ret = video_capture.grab()
            continue
        counter += 1

    ret, frame = video_capture.read()
    if ret:
        frame = image_resize(frame, dim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        n_of_frame += 1
        # if render:
        #     video_results.write(frame)
        # cv2.imshow("frame", frame)
    else:
        break
    waiting_time = int(1000 / fps)
    key = cv2.waitKey(1)
    if key == ord("x"):
        break


video_capture.release()
cv2.destroyAllWindows()
