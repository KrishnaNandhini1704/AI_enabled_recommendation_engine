import os
import requests

def download_file(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    dest_dir = os.path.join("data", "raw")
    dest_path = os.path.join(dest_dir, "online_retail_II.xlsx")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        download_file(url, dest_path)
    except Exception as e:
        print(f"Error downloading file: {e}")
