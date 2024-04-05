from utils import (read_video, save_video)
from trackers import PlayerTracker, ballTracker
from court_line_detector import CourtLineDetector
import cv2

def main():
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect player and ball
    player_tracker = PlayerTracker('yolov8x')
    ball_tracker = ballTracker(model_path="models/yolo5_last.pt")

    player_detection = player_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                    stub_path="tracker_stubs/player_detection.pkl")
    ball_detection = ball_tracker.detect_frames(video_frames,
                                                read_from_stub=True,
                                                stub_path="tracker_stubs/ball_detection.pkl")
    ball_detection = ball_tracker.interpolate_ball_position(ball_detection)

    # Court Line Detector Model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #Choose Player
    player_detection = player_tracker.choose_and_filter_players(court_keypoints, player_detection)

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detection)

    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()