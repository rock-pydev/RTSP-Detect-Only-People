from imageai.Detection import VideoObjectDetection
import cv2
import os


PERCENT = 0.8

def detect():
    """
    Count number of people in test video per second
    """
    execution_path = os.getcwd()

    # Add your camera credential
    # camera = cv2.VideoCapture('rtsp://username:password@0.0.0.0')

    # Test Video
    camera = cv2.VideoCapture('test.mp4')

    def getSizeVD(vcap):
        """
        Get Size of Camera Frame
        """
        if vcap.isOpened():
            width  = vcap.get(3)
            height = vcap.get(4)
            return width*height

    video_size = getSizeVD(camera)

    def forFrame(frame_number, output_array, output_count):
        """
        Get size of person on frame
        """
        size_list = []
        for item in output_array:
            if item.get('name', '') is 'person':
                if item.get('box_points', []):
                    [x1,y1,x2,y2] = item.get('box_points', [])
                    size = (x2 - x1) * (y2 - y1)
                    size_list.append(size)
        if size_list:
            if max(size_list) > video_size * PERCENT:
                # Add your shell script
                print("Detected Closed Person")

    # Create Video Detection Module based on Yolo
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    video_detector.loadModel()

    video_detector.detectObjectsFromVideo(
        camera_input=camera,
        output_file_path=os.path.join(execution_path, "test"),
        # save_detected_video=False,
        frames_per_second=10,
        per_frame_function=forFrame,
        minimum_percentage_probability=30
    )

if __name__ == "__main__":
    detect()