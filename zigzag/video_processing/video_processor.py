# myapp/process_video.py

from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from .utils import TimeEstimator, TimeAnnotator
from tqdm.notebook import tqdm
import numpy as np
import cv2
from ultralytics import YOLO
import time

MODEL_POSE = "yolov8x-pose.pt"
model_pose = YOLO(MODEL_POSE)

MODEL = "yolov8x.pt"
model = YOLO(MODEL)

# dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# get class id ball and bottle
CLASS_ID = [32, 39, 0]
PLAYER_CLASS_ID = 0
BOTTLE_CLASS_ID = 39
BALL_CLASS_ID = 32

def process_video(source_video_path, target_video_path):
    source_video_path = source_video_path.strip()
    target_video_path = target_video_path.strip()

    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(source_video_path)
    # create frame generator
    generator = get_video_frames_generator(source_video_path)
    # create instance of BoxAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
    # create TimeEstimator instance
    time_estimator = TimeEstimator()
    start_time = None
    end_time = None
    time_taken = None
    found_nearest_bottle = False

    try:
        # open target video file
        with VideoSink(target_video_path, video_info) as sink:
            for frame in tqdm(generator, total=video_info.total_frames):
                results_poses = model_pose.track(frame, persist=True)
                annotated_frame = results_poses[0].plot()

                results_objects = model.track(frame, persist=True, conf=0.1)
                tracker_ids = results_objects[0].boxes.id.int().cpu().numpy() if results_objects[0].boxes.id is not None else None
                detections = Detections(
                    xyxy=results_objects[0].boxes.xyxy.cpu().numpy(),
                    confidence=results_objects[0].boxes.conf.cpu().numpy(),
                    class_id=results_objects[0].boxes.cls.cpu().numpy().astype(int),
                    tracker_id=tracker_ids
                )

                # filter detections for bottles
                mask_bottles = np.array([class_id == BOTTLE_CLASS_ID for class_id in detections.class_id], dtype=bool)
                bottle_detections = Detections(
                    xyxy=detections.xyxy[mask_bottles],
                    confidence=detections.confidence[mask_bottles],
                    class_id=detections.class_id[mask_bottles],
                    tracker_id=detections.tracker_id[mask_bottles]
                )

                # Get keypoints for the person
                mask_player = np.array([class_id == PLAYER_CLASS_ID for class_id in detections.class_id], dtype=bool)
                player_detections = Detections(
                    xyxy=detections.xyxy[mask_player],
                    confidence=detections.confidence[mask_player],
                    class_id=detections.class_id[mask_player],
                    tracker_id=detections.tracker_id[mask_player]
                )
                keypoints = player_detections.xyxy

                if len(keypoints) > 0:
                    for player in keypoints:
                        player_position = int((player[0] + player[2]) / 2)  # Assuming keypoint[0][0] is the player's center point
                        print(f"Player found at {player_position}")

                        if len(bottle_detections.xyxy) >= 2 and not found_nearest_bottle:
                            nearest_bottle = time_estimator.set_start_bottle(player_position, bottle_detections)
                            start_x = time_estimator.update_bottle_position(nearest_bottle, bottle_detections)
                            found_nearest_bottle = True
                            print(f"Nearest bottle found at {start_x}")

                        # Check if crossed the line and get start timer
                        if time_estimator.crossed_start_bottle_line(player_position) and start_time is None and found_nearest_bottle:
                            start_time = time.time()
                            print(f"Started at {start_time}")
                            print("Timer has been started")

                        # Check if crossed the line and get end time
                        if time_estimator.crossed_start_bottle_line(player_position) and time.time() > start_time + 5 and found_nearest_bottle:
                            end_time = time.time()
                            time_taken = (end_time - start_time) / 10
                            print(f"Stopped at {end_time}")
                            print(f"Timer has been stopped after {time_taken} seconds!")
                            break
                        cv2.circle(annotated_frame, (player_position, 0), 20, (255, 0, 200), 20)

                n_bottle_x = time_estimator.update_bottle_position(nearest_bottle, bottle_detections) if found_nearest_bottle else None
                if n_bottle_x is not None:
                    start_x = n_bottle_x
                    time_estimator.set_start_line_position(start_x)
                cv2.circle(annotated_frame, (int(start_x), 0), 20, (255, 200, 0), 20)

                labels = [
                    f"id:{track_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    for i, (xyxy, confidence, class_id, track_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id))
                ]
                annotated_frame = box_annotator.annotate(frame=annotated_frame, detections=detections, labels=labels)
                time_annotator = TimeAnnotator(start_time, end_time)
                time_annotator.annotate(annotated_frame, time_estimator)

                # Display frame (uncomment to enable display during processing)
                # cv2.imshow("YOLOv8 Inference", annotated_frame)
                sink.write_frame(annotated_frame)

                # Break the loop if 'q' is pressed (uncomment to enable stopping with 'q')
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break

        if time_taken is not None:
            return time_taken
        else:
            print("The player did not return to the start bottle")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()

# Example usage:
# SOURCE_VIDEO_PATH = "dataset/running.mp4"
# TARGET_VIDEO_PATH = "dataset/running_result.mp4"
# time_taken = process_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH)
# print(f"Time taken: {time_taken}")
