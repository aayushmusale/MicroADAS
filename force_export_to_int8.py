import os

# Protect against the TensorFlow/Keras environment clash
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from ultralytics import YOLO

def generate_production_binary():
    print("Loading your trained weights...")
    # UPDATE THIS PATH if your file is in train2, train3, etc.
    model = YOLO(r"runs\detect\train\weights\best.pt") 

    print("Compressing to INT8 TFLite (This takes about 60 seconds)...")
    exported_path = model.export(
        format="tflite", 
        int8=True, 
        imgsz=320,
        # data="data.yaml"
    )
    
    print(f"\nSUCCESS! Your final model is located at: {exported_path}")

if __name__ == "__main__":
    generate_production_binary()