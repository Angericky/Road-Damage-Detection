#python3 detect.py --weights weights/IMSC/last_100_100_640_16.pt weights/IMSC/last_95_640_16.pt weights/IMSC/last_120_640_32_aug2.pt --img 640 --source /dev/disk2/zjj/data/abnormal_detection/test/IMG --conf-thres 0.05 --iou-thres 0.3 --agnostic-nms
python3 detect.py --weights runs/exp28/weights/best.pt --img 1080 --source /dev/disk2/zjj/data/abnormal_detection/Part2/IMG --conf-thres 0.20 --iou-thres 0.3 --agnostic-nms --output inference/pred_img_1080/
