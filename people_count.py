from imageai.Detection import VideoObjectDetection
import cv2
import os

def detect():
    execution_path = os.getcwd()

    def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
        print(average_output_count.get('person', 0))


    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    video_detector.loadModel()

    video_detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, "test.mp4"),
        output_file_path=os.path.join(execution_path, "traffic_detected"),
        frames_per_second=10,
        per_second_function=forSeconds,
        minimum_percentage_probability=30
    )

if __name__ == "__main__":
    detect()