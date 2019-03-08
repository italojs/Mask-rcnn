# MASK-RCNN implementation

 It's a script to generate "AI buzzword images" 

## Install
```
pip install -r requirements.txt
```
## Image

![](images/output.png)

### Basic usage
```
python3 generate-image.py --classes_path <path>/object_detection_classes_coco.txt --inception <path>/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt --frozen_inference <path>/frozen_inference_graph.pb --image_path <path>/image.jpeg 
``` 

## Full usage
```
python3 generate-image.py --classes_path <path>/object_detection_classes_coco.txt --inception <path>/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt --frozen_inference <path>/frozen_inference_graph.pb --image_path <path>/image.jpeg --confidence 0.65 --threshold 0.3
```

### Parameters
To know more about parameters use `python generate-image.py -h`:

```
python3 generate-image.py -h
usage: generate-image.py [-h] -i IMAGE_PATH [-m VIDEO_PATH] -p CLASSES_PATH -e
                         INCEPTION -f FROZEN_INFERENCE [-o OUTPUT]
                         [-c CONFIDENCE] [-t THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_PATH, --image_path IMAGE_PATH
                        path to input image
  -m VIDEO_PATH, --video_path VIDEO_PATH
                        path to input video
  -p CLASSES_PATH, --classes_path CLASSES_PATH
                        path to object_detection_classes_coco.txt
  -e INCEPTION, --inception INCEPTION
                        path to mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
  -f FROZEN_INFERENCE, --frozen_inference FROZEN_INFERENCE
                        path to frozen_inference_graph.pb
  -o OUTPUT, --output OUTPUT
                        path to frozen_inference_graph.pb
  -c CONFIDENCE, --confidence CONFIDENCE
                        minimum probability to filter weak detections
  -t THRESHOLD, --threshold THRESHOLD
                        minimum threshold for pixel-wise mask segmentation
```

## Video

### Basic usage
```
python3 generate-video.py --classes_path <path>/object_detection_classes_coco.txt --inception <path>/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt --frozen_inference <path>/frozen_inference_graph.pb --video_path <path>/video.mp4 --output <path>/video-output.mp4 
``` 

your videos will be processed and saved in output path

## Full usage
```
python3 generate-video.py --classes_path <path>/object_detection_classes_coco.txt --inception <path>/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt --frozen_inference <path>/frozen_inference_graph.pb --video_path <path>/video.mp4 --output <path>/video-output.mp4 --confidence 0.65 --threshold 0.3
``` 

### Parameters
To know more about parameters use `python3 generate-video.py -h`:

```
usage: generate-video.py [-h] [-i IMAGE_PATH] -m VIDEO_PATH -p CLASSES_PATH -e
                         INCEPTION -f FROZEN_INFERENCE -o OUTPUT
                         [-c CONFIDENCE] [-t THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_PATH, --image_path IMAGE_PATH
                        path to input image
  -m VIDEO_PATH, --video_path VIDEO_PATH
                        path to input video
  -p CLASSES_PATH, --classes_path CLASSES_PATH
                        path to object_detection_classes_coco.txt
  -e INCEPTION, --inception INCEPTION
                        path to mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
  -f FROZEN_INFERENCE, --frozen_inference FROZEN_INFERENCE
                        path to frozen_inference_graph.pb
  -o OUTPUT, --output OUTPUT
                        path to frozen_inference_graph.pb
  -c CONFIDENCE, --confidence CONFIDENCE
                        minimum probability to filter weak detections
  -t THRESHOLD, --threshold THRESHOLD
                        minimum threshold for pixel-wise mask segmentation
```

