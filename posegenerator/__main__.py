import click
import cv2
import pathlib
import json
import sys
import numpy as np
import mediapipe as mp


# Credit to pysource.com
# https://pysource.com/2021/05/21/blur-faces-in-real-time-with-opencv-mediapipe-and-python/
class FaceLandmarks:
    def __init__(self):
        # Init the MediaPipe face detection library
        self.mp_face_detection = mp.solutions.face_detection
        # model_selection = 0 is for close (<2m away) faces, while 1 is for far (~5m away) faces
        # I figure most videos will want the "far" setting
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def get_facial_landmarks(self, frame):
        # Initialize frame properties, run face detection
        frame_height, frame_width, _ = frame.shape
        face_detection_results = self.face_detection.process(frame[:,:,::-1])

        # If a face is detected
        if face_detection_results.detections:
            # From all detected faces, pick the one with the highest confidence
            # This will likely be the subject of the video
            max_conf = 0
            for face_no, face in enumerate(face_detection_results.detections):
                conf = face.score[0]
                if conf > max_conf:
                    max_conf = conf
                    face_index = face_no

            # Get bounding box coordinates from data
            face_data = face_detection_results.detections[face_index].location_data

            box = face_data.relative_bounding_box

            # Mediapipe returns coordinates as a fraction of the frame size, so we need to convert for opencv
            xmin = round(box.xmin * frame_width)
            ymin = round(box.ymin * frame_height)
            width = round(box.width * frame_width)
            height = round(box.height * frame_height)

            start_point = (xmin, ymin)
            end_point = (start_point[0] + width, start_point[1] + height)

        # Return the bounding box coords
        return (start_point, end_point)

@click.command()
@click.argument('INPUT_VIDEO', type=click.Path(exists=True))
@click.argument('POSES_JSON', type=click.Path(exists=True))
@click.argument('OUTPUT_VIDEO', type=click.Path())
@click.option('-v', '--verbose',
              is_flag=True,
              help="Print more output.")
@click.option('-b', '--blur',
              is_flag=True,
              help="Blur faces.")
@click.option('-u', '--upper',
              is_flag=True,
              help="Only draw skeleton from the waist up.")
def main(input_video, poses_json, output_video, verbose, blur, upper):

    # Default values for body parts, connections, and color.
    # TODO: Read in these parameters from a JSON file
    bodyparts = ["head", "lankle", "lelbow", "lhip", "lknee", "lshoulder", "lwrist",
                 "pelvis", "rankle", "relbow", "rhip", "rknee", "rshoulder", "rwrist",
                 "thorax", "upperneck"]

    # Connections between body parts; numbers represent index in the bodyparts array
    connections = [ (8 , 11),
                    (11, 10),
                    (10, 7 ),
                    (7 , 3 ),
                    (3 , 4 ),
                    (4 , 1 ),
                    (10, 12),
                    (3 , 5 ),
                    (5 , 14),
                    (14, 12),
                    (5 , 2 ),
                    (2 , 6 ),
                    (12, 9 ),
                    (9 , 13),
                    (14, 15),
                    (15, 0)   ]

    # Init coordinate and line color arrays
    coords = np.empty(len(bodyparts), dtype=object)
    coords.fill((0,0))

    defaultColor = (27, 144, 254) # Bright orange
    colors = np.empty(len(connections), dtype=object)
    colors.fill(defaultColor)

    # Designate index of parts to NOT render if upper is selected
    # lankle, lknee, rankle, rknee by default
    lower_joints = [1, 4, 8, 11]

    # Load in the poses JSON file
    try:
        with open(str(poses_json),
                  "r", encoding="utf-8") as poses:
            pose_data = json.load(poses)
    except FileNotFoundError:
        print("Pose JSON file not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error reading pose JSON. Is it properly formatted?")
        sys.exit(1)

    # Read in the input video
    input = cv2.VideoCapture(input_video)

    # Get relevant metadata about video
    frame_width = int(input.get(3))
    frame_height = int(input.get(4))
    fps = input.get(cv2.CAP_PROP_FPS)

    # Prepare output video for writing
    extension = output_video.split('.')[-1]
    
    if extension == "webm":
        output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('V','P','8','0'), fps, (frame_width,frame_height))
    else:
        output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

    # Check for valid input/output
    if (input.isOpened()== False):
        print("Error opening input video.")
        sys.exit(1)

    if (output.isOpened()== False):
        print("Error opening output video.")
        sys.exit(1)

    # Used to track current frame and progress
    frameCount = 0
    totalFrames = int(input.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize landmarks for face blur
    if blur:
        fl = FaceLandmarks()

    # Loop through every frame of the input video
    while(input.isOpened() and frameCount < totalFrames):
        ret, frame = input.read()

        # If a valid frame
        if ret == True:
            if frameCount <= totalFrames and verbose:
                print("Processing frame " + str(1+frameCount) + " of " + str(totalFrames), end="")

            # Capture coordinates of each body part from JSON file for current frame
            for i in range(len(bodyparts)):
                x = int(pose_data[frameCount][bodyparts[i]]["coords"][0])
                y = int(pose_data[frameCount][bodyparts[i]]["coords"][1])
                coords[i] = (x,y)

            if verbose: print(".",end="")
            
            # Blur face before drawing lines
            if blur:
                face_bounding_box = fl.get_facial_landmarks(frame)

                blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
                mask = cv2.rectangle(frame, face_bounding_box[0], face_bounding_box[1], (255, 255, 255), -1)
                frame = np.where(mask==np.array([255, 255, 255]), blurred_frame, frame)

            # Connect joints according to connections array
            for i in range(len(connections)):
                
                # Get confidence level from both endpoints of the connection
                conf1 = pose_data[frameCount][bodyparts[connections[i][0]]]["pointEstimationConfidence"][0]
                conf2 = pose_data[frameCount][bodyparts[connections[i][1]]]["pointEstimationConfidence"][0]
                
                # Only draw line if both endpoints have a high enough confidence level
                if conf1 >= 0.1 and conf2 >= 0.1:
                    x1 = coords[connections[i][0]][0]
                    y1 = coords[connections[i][0]][1]

                    x2 = coords[connections[i][1]][0]
                    y2 = coords[connections[i][1]][1]
                    
                    if not upper or (upper and connections[i][0] not in lower_joints
                                           and connections[i][1] not in lower_joints):
                        cv2.line(frame, (x1, y1), (x2, y2), colors[i], 5, lineType=cv2.LINE_AA)

            if verbose: print(".",end="")

            # Draw circles at each joint
            for i in range(len(coords)):
                # Get confidence level for current joint
                conf = pose_data[frameCount][bodyparts[i]]["pointEstimationConfidence"][0]
                if conf >= 0.1:
                    if not upper or (upper and i not in lower_joints):
                        cv2.circle(frame, (coords[i][0], coords[i][1]), 5, (201, 91, 0), -1, lineType=cv2.LINE_AA)

            # Write output frame
            output.write(frame)

            if verbose: print(".done", end='\r')

        else:
            # Break loop when done with the input video
            break
        frameCount += 1

    print("Processed " + str(totalFrames) + " frames. Output can be found in " + str(output_video))

    # Release the video objects
    input.release()
    output.release()

if __name__ == "__main__":
    main()
