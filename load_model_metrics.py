from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/train/weights/best.pt")

    metrics = model.val(data='data.yaml')           # testing the model against testing images
    print(f"Final mAP50-95 (Overall Score): {metrics.box.map:.3f}") 
    print(f"Final Precision: {metrics.box.mp:.3f}") 
    print(f"Final Recall: {metrics.box.mr:.3f}") 


if __name__ == "__main__":
    main()