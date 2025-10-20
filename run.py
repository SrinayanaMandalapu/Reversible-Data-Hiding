# run.py
import os
import csv
from gap import demo_gap

def process_dataset(dataset_path, csv_path="metrics.csv"):
    rows = []

    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm')):
                image_path = os.path.join(root, fname)
                rel_path = os.path.relpath(image_path, dataset_path)

                print(f"Processing: {image_path}")
                try:
                    res = demo_gap(image_path, enc_key="testkey", show=False)
                    rows.append({
                        "image_path": rel_path,
                        "height": res["height"],
                        "width": res["width"],
                        "aux_len": res["aux_len"],
                        "used_bits": res["used_bits"],
                        "capacity_bits": res["capacity_bits"],
                        "capacity_bpp": f'{res["capacity_bpp"]:.6f}',
                        "exact_reconstruction": int(res["exact_reconstruction"]),
                        "mse": f'{res["mse"]:.10f}',
                        "psnr_db": ('inf' if res["psnr_db"] == float("inf") else f'{res["psnr_db"]:.6f}'),
                        "time_total_s": f'{res["time_total_s"]:.6f}',
                    })
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    if rows:
        fieldnames = [
            "image_path","height","width",
            "aux_len","used_bits",
            "capacity_bits","capacity_bpp",
            "exact_reconstruction","mse","psnr_db",
            "time_total_s"
        ]
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"\nSaved metrics CSV -> {os.path.abspath(csv_path)}")
        total = len(rows)
        exact = sum(r["exact_reconstruction"] for r in rows)
        print(f"Processed {total} images.")
        print(f"Perfect reconstruction: {exact}/{total} ({(exact/total)*100:.2f}%)")
    else:
        print("No images found / processed; nothing to write.")

if __name__ == "__main__":
    dataset_path = "BOSSbase_1.01"
    process_dataset(dataset_path, csv_path="metrics.csv")
