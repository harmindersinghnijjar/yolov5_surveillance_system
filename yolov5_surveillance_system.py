# Import necessary modules
import os  # Operating system operations like file manipulation
import subprocess  # To run shell commands
import time  # To use sleep function and get current time
import datetime  # To format time
import torch  # PyTorch, for loading the YOLO model
import platform  # To identify the operating system
import ffmpeg  # To handle video conversion
from PIL import Image  # To handle image files
import shutil  # To perform high level file operations
import glob  # To find all the pathnames matching a specified pattern
import threading  # To perform some tasks in background (like recording video)
import logging  # To log the activities of the program

# Start logging process
logger = logging.getLogger(__name__)  # Get a logger object, it'll collect logs
logger.setLevel(logging.INFO)  # Set the level to gather only info and above level logs

# Create a handler that writes log messages to a file
file_handler = logging.FileHandler('log.log', mode='a')  # 'a' means logs will be appended to the same file
log_format = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] [%(pathname)s:%(lineno)d] - %(message)s - '
                               '[%(process)d:%(thread)d]')
file_handler.setFormatter(log_format)  # Set the format of the log
logger.addHandler(file_handler)  # Add the file handler to the logger

# Create a handler that writes log messages to the console
console_handler = logging.StreamHandler()  # Console handler
console_handler.setFormatter(log_format)  # Set the format
logger.addHandler(console_handler)  # Add console handler to logger

# Short-hand for logging
print = logger.info  # Now print() will behave as logger.info()

# Function to clear the console, helps in keeping the console clean
def clear_console():
    # Check if the OS is windows or others
    # 'cls' clears console in windows and 'clear' does the same in linux/mac
    command = 'cls' if platform.system().lower() == 'windows' else 'clear'
    os.system(command)  # Execute the command (clear console)

# Class containing methods related to YOLOv5 model
class YOLOv5Inference:
    # Initialize the class with the model
    def __init__(self, model_path):
        try:
            print('Loading YOLOv5 model...')
            self.model = torch.hub.load('ultralytics/yolov5', model_path)  # Load the model
            print('YOLOv5 model loaded successfully.')
        except Exception as e:  # Catch any error occurred during model loading
            logger.error(f"Failed to load model: {e}")  # Log the error
            raise e  # Raise the error further

    # Capture image using the camera
    def capture_image(self, image_path):
        try:
            print('Capturing image...')
            libcamera_command = f"libcamera-still -o {image_path}"  # Command to capture image
            subprocess.run(libcamera_command, shell=True)  # Run the command
            print(f'Image captured and saved at {image_path}.')
        except Exception as e:  # Catch any error occurred during image capturing
            logger.error(f"Failed to capture image: {e}")  # Log the error
            raise e  # Raise the error further

    # Detect objects in the image
    def detect_objects(self, image_path):
        try:
            print('Detecting objects in the image...')
            results = self.model(image_path)  # Run the model on the image
            print('Object detection completed.')
            return results  # Return the results
        except Exception as e:  # Catch any error occurred during object detection
            logger.error(f"Failed to detect objects: {e}")  # Log the error
            raise e  # Raise the error further

    # Record video using the camera
    def record_video(self, video_path):
        try:
            print('Recording video...')
            record_command = f"libcamera-vid -t 30000 -o {video_path}"  # Command to record video
            subprocess.run(record_command, shell=True)  # Run the command
            avi_video_path = video_path.replace('.mp4', '.avi')  # Change the extension to .avi
            ffmpeg.input(video_path).output(avi_video_path).run()  # Convert the video to .avi format
            os.remove(video_path)  # Remove the original .mp4 video
            print(f'Video recorded and saved at {avi_video_path}.')
            return avi_video_path  # Return the .avi video path
        except Exception as e:  # Catch any error occurred during video recording
            logger.error(f"Failed to record video: {e}")  # Log the error
            raise e  # Raise the error further

    # Save the detection results
    def save_inference(self, results, output_dir):
        try:
            print('Saving detection results...')
            df = results.pandas().xyxy[0]  # Get the results in pandas DataFrame format
            df.to_csv(os.path.join(output_dir, 'inference_results.csv'), index=False)  # Save the DataFrame to a .csv file
            img_with_boxes = Image.fromarray(results.render()[0])  # Get the image with bounding boxes
            img_path = os.path.join(output_dir, 'image_with_boxes.jpg')  # Path to save the image
            img_with_boxes.save(img_path)  # Save the image
            print(f'Detection results saved at {output_dir}.')
        except Exception as e:  # Catch any error occurred during saving detection results
            logger.error(f"Failed to save inference results: {e}")  # Log the error
            raise e  # Raise the error further

# Function to delete oldest files if number of files exceeds max limit
def cleanup_files(source_dir, max_files):
    try:
        print('Cleaning up excess files...')
        file_list = sorted(glob.glob(os.path.join(source_dir, '*')), key=os.path.getmtime)  # List of all files sorted by modification time
        while len(file_list) > max_files:  # While number of files exceeds max limit
            os.remove(file_list[0])  # Remove the oldest file
            file_list = file_list[1:]  # Update the file list
        print('File cleanup completed.')
    except Exception as e:  # Catch any error occurred during file cleanup
        logger.error(f"Failed to cleanup files: {e}")  # Log the error
        raise e  # Raise the error further

# Function to delete oldest folders if number of folders exceeds max limit
def cleanup_folders(source_dir, max_folders):
    try:
        print('Cleaning up excess folders...')
        folder_list = sorted([d for d in glob.glob(os.path.join(source_dir, '*')) if os.path.isdir(d)], key=os.path.getmtime)  # List of all folders sorted by modification time
        while len(folder_list) > max_folders:  # While number of folders exceeds max limit
            shutil.rmtree(folder_list[0])  # Remove the oldest folder
            folder_list = folder_list[1:]  # Update the folder list
        print('Folder cleanup completed.')
    except Exception as e:  # Catch any error occurred during folder cleanup
        logger.error(f"Failed to cleanup folders: {e}")  # Log the error
        raise e  # Raise the error further

# Function to record video in the background
def record_video_in_background(video_path, detected_person_dir, timestamp):
    try:
        print('Recording video in the background...')
        record_command = f"libcamera-vid -t 30000 -o {video_path}"  # Command to record video
        subprocess.run(record_command, shell=True)  # Run the command
        avi_video_path = video_path.replace('.mp4', '.avi')  # Change the extension to .avi
        ffmpeg.input(video_path).output(avi_video_path).run()  # Convert the video to .avi format
        os.remove(video_path)  # Remove the original .mp4 video
        shutil.move(avi_video_path, os.path.join(detected_person_dir, f"{timestamp}.avi"))  # Move the .avi video to the person detected folder
        print(f'Background video recording completed and moved to {detected_person_dir}.')
    except Exception as e:  # Catch any error occurred during background video recording
        logger.error(f"Failed to record video in background: {e}")  # Log the error
        raise e  # Raise the error further

# Main program
if __name__ == "__main__":
    try:
        yolov5 = YOLOv5Inference('yolov5s')  # Load the YOLOv5 model
        while True:  # Keep running the program forever
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Get the current timestamp
            base_dir = "/home/admin/YOLOv5_Source"  # Base directory where image/video will be saved
            result_dir = os.path.join("/home/admin/YOLOv5_Results", timestamp)  # Directory where results will be saved
            os.makedirs(result_dir, exist_ok=True)  # Create the directory if doesn't exist
            image_path = os.path.join(base_dir, f"{timestamp}.jpg")  # Path where image will be saved
            video_path = os.path.join(base_dir, f"{timestamp}.mp4")  # Path where video will be saved
            yolov5.capture_image(image_path)  # Capture the image
            cleanup_files(base_dir, 20)  # Clean up excess files
            clear_console()  # Clear the console
            results = yolov5.detect_objects(image_path)  # Detect objects in the image
            yolov5.save_inference(results, result_dir)  # Save the detection results
            cleanup_files(result_dir, 20)  # Clean up excess files
            detected_persons = results.pandas().xyxy[0]['name'].values  # Get the names of all detected objects
            if 'person' in detected_persons:  # If a person is detected
                detected_person_dir = os.path.join("/home/admin/Detected_Persons", timestamp)  # Directory to save results when a person is detected
                os.makedirs(detected_person_dir, exist_ok=True)  # Create the directory if doesn't exist
                video_thread = threading.Thread(target=record_video_in_background, args=(video_path, detected_person_dir, timestamp))  # Start a thread to record video in the background
                video_thread.start()  # Start the thread
                time.sleep(30)  # Wait for 30 seconds
                shutil.move(image_path, os.path.join(detected_person_dir, f"{timestamp}.jpg"))  # Move the image to the person detected folder
            cleanup_folders("/home/admin/YOLOv5_Results", 20)  # Clean up excess folders
            time.sleep(5)  # Wait for 5 seconds before the next iteration
    except Exception as e:  # Catch any error occurred during the main program
        logger.error(f"An error occurred: {e}")  # Log the error
