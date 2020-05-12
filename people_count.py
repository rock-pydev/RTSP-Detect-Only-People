from imageai.Detection import VideoObjectDetection
import cv2
import os

LIMIT = 10

def detect():
    """
    Count number of people in test video per second
    """
    execution_path = os.getcwd()

    def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
        if average_output_count.get('person', 0) == LIMIT:
            # Add module for calling external API
            pass
    
    # Video from Camera
    camera = cv2.VideoCapture('rtsp://username:password@0.0.0.0')

    # Create Video Detection Module based on Yolo
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    video_detector.loadModel()

    video_detector.detectObjectsFromVideo(
        # camera_input=camera,
        input_file_path=os.path.join(execution_path, "test.mp4"),
        save_detected_video=False,
        frames_per_second=10,
        per_second_function=forSeconds,
        minimum_percentage_probability=30
    )

if __name__ == "__main__":
    detect()