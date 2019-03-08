import os
import cv2
import time
import mask_rcnn

args = mask_rcnn.get_args()

image = cv2.imread(args["image_path"])
labels = mask_rcnn.get_labels(args["classes_path"])

nn = mask_rcnn.get_net( args["frozen_inference"], args["inception"])

start = time.time()
(boxes, masks) = mask_rcnn.process_image(nn,image)
end = time.time()

print("[INFO] {:.2f} seconds to predict {} masks".format(end - start, len(masks)))

drawed_image = mask_rcnn.draw_boxes_masks(
    image,
    boxes,
    masks,
    labels,
    float(args["confidence"]),
    float(args["threshold"])
)

cv2.imshow("Output", drawed_image)
cv2.waitKey(0)