from typing import Any
import streamlit as st
import cv2
import numpy as np
from cvlib.object_detection import ObjectDetection
from cvlib.tracker import EuclideanDistTracker
import os
import time


def image_resize(img, dim):
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


class_ref = ["car", "motorbike"]

color_ref = [(0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]
st.set_page_config(page_title="Vehicle Counting", page_icon="🚗", layout="wide")
st.title("Vehicle Search")

st.markdown("This is a demo for vehicle counting")
video = None
take_class = None
video_folder = "./Videos"


class CancleState:
    def __init__(self):
        self._state = False

    def set(self):
        self._state = True

    def get(self):
        return self._state

    def reset(self):
        self._state = False
        return self._state

    def __call__(self) -> Any:
        return self._state


with st.form("InputForm"):
    st.subheader("Choose video, you need to put the video in Videos folder")
    video_list = os.listdir(video_folder)
    video = st.selectbox("Video", video_list)
    st.subheader("Model")
    model_list = ["yolov5n", "yolov8n", "rtdetr"]
    model_chose = st.selectbox("Model", model_list)
    st.subheader("Parameters")
    conf, iou = st.columns(2)
    conf_val = conf.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01
    )
    iou_val = iou.slider(
        "IoU threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.01
    )
    st.subheader("Image resize")
    # input width and height on the same row
    widthc, heightc = st.columns(2)
    width = widthc.number_input(
        "Width", min_value=0, max_value=10000, value=1280, step=1
    )
    height = heightc.number_input(
        "Height", min_value=0, max_value=10000, value=720, step=1
    )

    st.subheader("Counting Parameters")
    a, d, l = st.columns(3)
    acceptable = a.number_input(
        "Acceptable", min_value=0, max_value=1000, value=20, step=1
    )
    delta = d.number_input("Delta", min_value=0, max_value=1000, value=25, step=1)
    liner = l.number_input("Line", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    st.subheader("Limit queue")
    st.caption("The limit queue to avoid duplicate count")
    limit_queue = st.number_input(
        "Limit queue", min_value=5, max_value=10000, value=100, step=1
    )
    st.subheader("Frame cut option")
    st.caption("Cut the frame to reduce the number of frame to process")
    frameCut = st.checkbox("Frame cut", value=False)
    fc, ds = st.columns(2)
    targer_counter = fc.number_input(
        "Skip n-th frame (ignore if framecut == false)",
        min_value=0,
        max_value=1000,
        value=1,
        step=1,
    )
    dis_scalar = ds.number_input(
        "Distance scalar", min_value=0, max_value=100, value=1, step=1
    )
    st.subheader("Cut the frame")
    st.caption("Cut the frame to this size to reduce time to process")
    cap_x, cap_y, cap_w, cap_h = st.columns(4)
    cap_x = cap_x.number_input("CX", min_value=0, max_value=10000, value=0, step=1)
    cap_y = cap_y.number_input("CY", min_value=0, max_value=10000, value=0, step=1)
    cap_w = cap_w.number_input(
        "CWidth", min_value=0, max_value=10000, value=1280, step=1
    )
    cap_h = cap_h.number_input(
        "CHeight", min_value=0, max_value=10000, value=720, step=1
    )
    st.subheader("Show capture")
    st.caption("Show the capture zone of the video")
    show_cap = st.checkbox("Show capture", value=True)

    st.subheader("Render")
    st.caption("Show video in processing")
    render = st.checkbox("Show", value=True)
    form_submitted = st.form_submit_button("Count", use_container_width=True)


def draw_with_alpha(img, draw_function, up, lp, color, width, alpha=0.5):
    overlay = np.zeros_like(img, dtype=np.uint8)
    draw_function(overlay, up, lp, color, width)
    mask = overlay.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)[mask]


if form_submitted:
    with st.spinner("Processing..."):
        canceller = CancleState()
        video_capture = cv2.VideoCapture(os.path.join(video_folder, video))
        dim = (width, height)
        detector = ObjectDetection(model=model_chose)
        tracker = EuclideanDistTracker(class_ref=class_ref, limit_queue=limit_queue)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        base_dis = delta
        dis_scalar = (targer_counter + 1) / targer_counter if frameCut else 1
        if not os.path.exists(f"Videos_ouput"):
            os.makedirs(f"Videos_ouput")
        video_results = cv2.VideoWriter(
            f"Videos_ouput/result.avi",
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps,
            (width, height),
        )
        line = int(cap_h * liner)
        ret, frame = video_capture.read()
        light_mean_value = np.mean(frame) / 255
        st.button("Cancel", on_click=canceller.set)
        st.text(f"{'Night' if light_mean_value < 0.3 else 'Day'}")
        frame_count, fps = st.empty(), st.empty()
        video_render = st.empty()
        progerss_bar = st.progress(0)
        counter = 0
        while True:
            if canceller():
                form_submitted = False
                canceller.reset()
                break
            if frameCut:
                if counter == targer_counter:
                    counter = 0
                    ret = video_capture.grab()
                    continue
                counter += 1
            ret, frame = video_capture.read()
            if ret:
                tic = time.time()
                frame = image_resize(frame, dim)
                roi = frame[cap_y : cap_y + cap_h, cap_x : cap_x + cap_w]
                results, classes, confs = detector.predict(
                    roi,
                    specific_class=take_class,
                    conf=conf_val,
                    iou=iou_val,
                )
                classes = [1 if c == 3 else 0 for c in classes]
                if show_cap:
                    # cv2.rectangle(frame, (cap_x, cap_y), (cap_x + cap_w, cap_y + cap_h), (255, 255, 0), 2)
                    cv2.line(roi, (0, line), (cap_w, line), (107, 214, 205), 2)
                    draw_with_alpha(
                        frame,
                        cv2.rectangle,
                        (cap_x, cap_y),
                        (cap_x + cap_w, cap_y + cap_h),
                        (255, 255, 0),
                        2,
                    )
                    # cv2.line(roi, (0, line+acceptable), (cap_w, line+acceptable), (214, 214, 214), 2)
                    # cv2.line(roi, (0, line-acceptable), (cap_w, line-acceptable), (214, 214, 214), 2)
                    draw_with_alpha(
                        roi,
                        cv2.line,
                        (0, line + acceptable),
                        (cap_w, line + acceptable),
                        (214, 214, 214),
                        2,
                    )
                    draw_with_alpha(
                        roi,
                        cv2.line,
                        (0, line - acceptable),
                        (cap_w, line - acceptable),
                        (214, 214, 214),
                        2,
                        0.8,
                    )

                if len(results) > 0:
                    presults = tracker.update(
                        results,
                        classes=classes,
                        liney=line,
                        acceptable=acceptable,
                        delta=base_dis * dis_scalar,
                    )
                    for i, package in enumerate(presults):
                        ux, uy, lx, ly, id, direction, clsid = package
                        if clsid == 0:
                            cv2.rectangle(roi, (ux, uy), (lx, ly), color_ref[clsid], 2)
                            cv2.putText(
                                roi,
                                f"id: {id} class: {class_ref[classes[i]]} conf: {confs[i]:0.2f}",
                                (ux, uy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color_ref[clsid],
                                2,
                            )
                cv2.putText(
                    frame,
                    f"Vehicle Count: {str(tracker.count['car'])}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Active Vehicle: {str(len(results))}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if render:
                    video_render.image(frame, channels="BGR")
                video_results.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                toc = time.time()
                time_taken = toc - tic
                progerss_bar.progress(
                    int(
                        video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                        / video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                        * 100
                    )
                )
                frame_count.text(
                    f"Frame: {video_capture.get(cv2.CAP_PROP_POS_FRAMES):0.0f}/{video_capture.get(cv2.CAP_PROP_FRAME_COUNT):0.0f} | FPS: {1/time_taken:0.0f}"
                )

            else:
                break
        video_capture.release()
        video_results.release()
        cv2.destroyAllWindows()
        st.success("Done!")
        st.balloons()
