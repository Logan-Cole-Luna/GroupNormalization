import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import openpyxl


# Function to evaluate the YOLO model
def evaluate_yolo_model(model, image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    processing_times = []
    inference_times = []
    num_boxes = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        start_time = time.time()
        results = model(image_path)
        end_time = time.time()

        result = results[0]
        boxes = result.boxes  # YOLO boxes object

        processing_time = end_time - start_time
        processing_times.append(processing_time)

        inference_times.append(result.times['inference'])
        num_boxes.append(len(boxes))

    # Calculate the average metrics
    avg_processing_time = np.mean(processing_times)
    avg_inference_time = np.mean(inference_times)
    avg_num_boxes = np.mean(num_boxes)

    return avg_processing_time, avg_inference_time, avg_num_boxes


# Load the YOLO model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Define the folder containing images
image_folder = "Drone1/img"

# Evaluate the YOLO model
processing_time, inference_time, num_boxes = evaluate_yolo_model(model, image_folder)

# Create a DataFrame to store the results
results_df = pd.DataFrame([{
    "Model": "YOLOv8n",
    "Processing Time (s)": processing_time,
    "Inference Time (s)": inference_time,
    "Average Number of Boxes": num_boxes
}])

# Print the results
print(results_df)

# Save the results to an Excel file
results_df.to_excel("yolo_model_evaluation_results.xlsx", index=False)

# Plot the results
metrics = ["Processing Time (s)", "Inference Time (s)", "Average Number of Boxes"]
results_df.set_index("Model", inplace=True)

for metric in metrics:
    plt.figure()
    results_df[metric].plot(kind='bar', title=metric)
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.savefig(f"{metric}.png")
    plt.show()

# Display all results in one graph
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(results_df[metric], marker='o', label=metric)
plt.title('YOLO Model Performance Comparison')
plt.ylabel('Score / Time')
plt.xlabel('Model')
plt.legend()
plt.grid(True)
plt.savefig("YOLO_Model_Performance_Comparison.png")
plt.show()
