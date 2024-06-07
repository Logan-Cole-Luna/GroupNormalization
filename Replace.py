import os
import re

# Directory containing the extracted files
extraction_dir = 'ultralytics/ultralytics'


# Function to replace BatchNorm2d with GroupNorm
def replace_batchnorm_with_groupnorm(file_path):
    with open(file_path, 'r') as file:
        file_data = file.read()

    # Replace BatchNorm2d with GroupNorm
    file_data = re.sub(r'nn\.BatchNorm2d\((\w+)\)', r'nn.GroupNorm(32, \1)', file_data)

    with open(file_path, 'w') as file:
        file.write(file_data)


# Traverse the directory and modify relevant files
for root, dirs, files in os.walk(extraction_dir):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            replace_batchnorm_with_groupnorm(file_path)
