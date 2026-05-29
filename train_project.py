import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"     # Force TensorFlow to use the legacy Keras architecture
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Suppress annoying warnings

from ultralytics import YOLO

def build_adas_obstacle_model():
    print("--- 1. Initializing YOLOv8 Nano ---")
    # Using the lightest model to ensure we hit the <2MB constraint
    model = YOLO("yolov8n.pt") 

    print("\n--- 2. Training Model ---")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=320,
        batch=64,
        patience=15,    # Added: Early stopping
        val=False,      # Added: Speeds up training time per epoch
        device="0"
    )

    # model.train(
    #     data="data.yaml",   # Points to your custom folder structure
    #     epochs=100,         # Adjust if it learns faster or slower
    #     imgsz=320,          # Keeps the model fast and small
    #     batch=64,
    #     device="0"            # Change to "0" if you have an NVIDIA GPU
    # )

    print("\n--- 3. Testing / Validation Metrics ---")
    # Evaluates the model against your images/testing directory
    metrics = model.val()           # testing the model against 177 testing images
    print(f"Final mAP50-95 (Overall Score): {metrics.box.map:.3f}") 
    print(f"Final Precision: {metrics.box.p.mean():.3f}") 
    print(f"Final Recall: {metrics.box.r.mean():.3f}") 

    print("\n--- 4. Visual Test & Latency Check ---")
    # Grab an image from your testing folder to see it in action
    # Change this filename to an actual image in your testing folder
    test_img_path = "dataset/images/testing/frames_0015.jpg" 
    
    if os.path.exists(test_img_path):
        results = model.predict(source=test_img_path, imgsz=320, show=True)
        speed_ms = results[0].speed['inference']
        print(f"Inference Latency on PC: {speed_ms:.2f} ms")
    else:
        print(f"Could not find {test_img_path} for visual test. Skipping.")

    print("\n--- 5. Exporting to Arduino-Compatible Format ---")
    # Exporting to TFLite INT8 to get the size under 2MB
    exported_path = model.export(
        format="tflite", 
        int8=True, 
        imgsz=320,
        # data="data.yaml"
    )
    
    print("\n--- 6. Final Model Size Check ---")
    if os.path.exists(exported_path):
        size_mb = os.path.getsize(exported_path) / (1024 * 1024)
        print(f"SUCCESS: Your INT8 TFLite model size is: {size_mb:.2f} MB")
        if size_mb <= 2.0:
            print("Status: Fits perfectly within hardware RAM constraints!")
        else:
            print("Status: WARNING - Model exceeds 2MB.")

if __name__ == "__main__":
    build_adas_obstacle_model()