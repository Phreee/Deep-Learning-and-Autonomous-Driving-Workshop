import csv
import os
import random
import urllib.request


RAW_BASE = "https://raw.githubusercontent.com/KansaiUser/BehavioralCloningTrackData/master"


def download_file(url, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    urllib.request.urlretrieve(url, dst_path)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "tutorial_runs", "behavioral_cloning_data")
    img_dir = os.path.join(out_dir, "IMG")
    os.makedirs(img_dir, exist_ok=True)

    csv_url = f"{RAW_BASE}/driving_log.csv"
    csv_path = os.path.join(out_dir, "driving_log.csv")
    download_file(csv_url, csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    # Sample a small subset for quick training
    random.seed(42)
    sample_size = 1000
    if len(reader) > sample_size:
        reader = random.sample(reader, sample_size)

    local_rows = []
    for row in reader:
        if not row:
            continue
        center_path = row[0]
        steering = row[3]
        filename = os.path.basename(center_path)
        raw_img_url = f"{RAW_BASE}/IMG/{filename}"
        local_img_path = os.path.join(img_dir, filename)
        download_file(raw_img_url, local_img_path)
        local_rows.append([local_img_path, steering])

    local_csv_path = os.path.join(out_dir, "driving_log_local.csv")
    with open(local_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "steering"])
        writer.writerows(local_rows)

    print(f"saved_csv {local_csv_path}")
    print(f"images {len(local_rows)}")


if __name__ == "__main__":
    main()
