import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import openpyxl


# Function to read ground truth from a file (assuming the format [x1, y1, x2, y2] per line)
def read_ground_truth(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ground_truths = []
    for line in lines:
        coords = list(map(int, line.strip().split()))
        ground_truths.append(coords)
    return ground_truths


# Function to evaluate the YOLO model
def evaluate_yolo_model(model, image_folder, ground_truth_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    processing_times = []
    inference_times = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        ground_truth_file = os.path.join(ground_truth_folder, os.path.splitext(image_file)[0] + '.txt')

        ground_truth = read_ground_truth(ground_truth_file)

        start_time = time.time()
        results = model(image_path)
        end_time = time.time()

        result = results[0]
        boxes = result.boxes  # YOLO boxes object
        preds = boxes.xyxy.cpu().numpy()

        processing_time = end_time - start_time
        processing_times.append(processing_time)

        # Calculate precision, recall, and F1 score for the current image
        if len(ground_truth) > 0 and len(preds) > 0:
            precision = precision_score(ground_truth, preds, average='macro')
            recall = recall_score(ground_truth, preds, average='macro')
            f1 = f1_score(ground_truth, preds, average='macro')
        else:
            precision, recall, f1 = 0, 0, 0

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        inference_times.append(result.times['inference'])

    # Calculate the average metrics
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1_scores)
    avg_processing_time = np.mean(processing_times)
    avg_inference_time = np.mean(inference_times)

    return avg_precision, avg_recall, avg_f1, avg_processing_time, avg_inference_time


# Load the YOLO model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Define the folders containing images and ground truths
image_folder = "Drone1/img"
ground_truth_folder = "Drone1/groundtruth_rect.txt"

# Evaluate the YOLO model
precision, recall, f1, processing_time, inference_time = evaluate_yolo_model(model, image_folder, ground_truth_folder)

# Create a DataFrame to store the results
results_df = pd.DataFrame([{
    "Model": "YOLOv8n",
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Processing Time (s)": processing_time,
    "Inference Time (s)": inference_time
}])

# Print the results
print(results_df)

# Save the results to an Excel file
results_df.to_excel("yolo_model_evaluation_results.xlsx", index=False)

# Plot the results
metrics = ["Precision", "Recall", "F1 Score", "Processing Time (s)", "Inference Time (s)"]
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
