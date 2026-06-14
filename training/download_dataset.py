import kagglehub

# Download latest version
path = kagglehub.dataset_download("bhaveshkumars/release-in-the-wild")

print("Path to dataset files:", path)