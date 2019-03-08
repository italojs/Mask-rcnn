# import the necessary packages
import os
import cv2
import utils
import time
import argparse
import mask_rcnn
import numpy as np

args = mask_rcnn.get_args(video=True)
labels = mask_rcnn.get_labels(args["classes_path"])
net = mask_rcnn.get_net( args["frozen_inference"], args["inception"])

vs = cv2.VideoCapture(args["video_path"])
writer = None

try:
	total = int(vs.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) if utils.is_cv2() \
		else int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	total = -1

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    start = time.time()
    (boxes, masks) = mask_rcnn.process_image(net,frame)
    end = time.time()

    drawed_image = mask_rcnn.draw_boxes_masks(
        frame,
        boxes,
        masks,
        labels,
        float(args["confidence"]),
        float(args["threshold"])
    )

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (drawed_image.shape[1], drawed_image.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] {:.2f} seconds to predict a unique frame".format(elap))
            print("[INFO] estimated total time to finish: {:.2f} seconds".format(
                (elap * total)/60))

    writer.write(drawed_image)
print("[INFO] cleaning up...")
writer.release()
vs.release()