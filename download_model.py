
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import os
import shutil

print("Downloading road damage AI model...")

weights_path = hf_hub_download(
    repo_id="nsr51324/Road_Damage_Object_Detection",
    filename="runs/detect/yolov8_road/weights/best.pt"
)

os.makedirs("models", exist_ok=True)

destination = os.path.join("models", "road_damage_best.pt")
shutil.copy(weights_path, destination)

print("\nModel downloaded successfully!")
print("Saved at:", destination)

# Load model and check its classes
model = YOLO(destination)

print("\nModel loaded successfully!")
print("Available detection classes:")
print(model.names)
