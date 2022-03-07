import click
import cv2
import pathlib
import json
import sys
import numpy as np

@click.command()
@click.argument('INPUT_VIDEO', type=click.Path(exists=True))
@click.argument('POSES_JSON', type=click.Path(exists=True))
@click.argument('OUTPUT_VIDEO', type=click.Path())
def main(input_video, poses_json, output_video):

    bodyparts = ["head", "lankle", "lelbow", "lhip", "lknee", "lshoulder", "lwrist",
                 "pelvis", "rankle", "relbow", "rhip", "rknee", "rshoulder", "rwrist",
                 "thorax", "upperneck"]

    coords = np.empty(16, dtype=object)
    coords.fill((0,0))

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

    defaultColor = (27, 144, 254)
    colors = np.empty(16, dtype=object)
    colors.fill(defaultColor)


    # Load in the poses json file
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

    input = cv2.VideoCapture(input_video)

    fps = int(input.get(2))
    frame_width = int(input.get(3))
    frame_height = int(input.get(4))

    fps = input.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

    if (input.isOpened()== False):
        print("Error opening input video.")
        sys.exit(1)

    if (output.isOpened()== False):
        print("Error opening output video.")
        sys.exit(1)

    frameCount = 0
    while(input.isOpened()):
        ret, frame = input.read()
        if ret == True:
            for i in range(len(bodyparts)):
                x = int(pose_data[frameCount][bodyparts[i]]["coords"][0])
                y = int(pose_data[frameCount][bodyparts[i]]["coords"][1])

                coords[i] = (x,y)

            # Connect joints
            for i in range(len(connections)):
                x1 = coords[connections[i][0]][0]
                y1 = coords[connections[i][0]][1]

                x2 = coords[connections[i][1]][0]
                y2 = coords[connections[i][1]][1]

                cv2.line(frame, (x1, y1), (x2, y2), colors[i], 5)

            # Draw circles at each joint
            for i in range(len(coords)):
                cv2.circle(frame, (coords[i][0], coords[i][1]), 5, (201, 91, 0), -1)

            output.write(frame)
        else:
            break
        frameCount += 1

    # Release the video objects
    input.release()
    output.release()

if __name__ == "__main__":
    main()
