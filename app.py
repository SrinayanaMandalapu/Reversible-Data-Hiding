import os
import csv
import time
import math
import cv2
import numpy as np
import gc
import imageio.v2 as imageio
import tensorflow as tf
from tensorflow.keras.models import load_model
from med import med_predictor_pixel, int_to_bits, bits_to_int

# med pipeline functions
from med import (
    label_map_generation, make_fixed_huffman_map, encode_label_map,
    encrypt_image, embed_aux_into_encrypted, decrypt_image,
    extract_aux_from_encrypted, decode_label_map_from_bits,
    recover_image_from_decrypted, compute_capacity
)
from cnn import train_cnn_model, cnn_predictor
from utils import classify_image


# ---------------- Environment Setup ----------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU for stability


# ---------------- Utility Functions ----------------
def evaluate_arrays(orig, rec):
    diff = orig.astype(float) - rec.astype(float)
    mse = float((diff ** 2).mean())
    psnr = float('inf') if mse == 0 else 10 * math.log10((255 ** 2) / mse)
    return mse, psnr


# ---------------- Full Processing Pipeline ----------------
def process_image_full_pipeline(image_path, model=None, predictor='MED', enc_key='testkey'):
    """Run full MED/CNN hybrid pipeline and perform correct diagnostics."""
    import os

    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)
    h, w = img.shape

    # 1) Label map generation
    if predictor == 'CNN' and model is not None:
        label, counts = label_map_generation(img, predictor='CNN', cnn_model=model)
    else:
        label, counts = label_map_generation(img, predictor='MED', cnn_model=None)

    # 2) Huffman mapping & encode label map
    mapping, rev = make_fixed_huffman_map(counts)
    aux_bits = encode_label_map(label, mapping)
    aux_len = len(aux_bits)

    # 3) Encrypt image (BEFORE embedding)
    encrypted, prng_mat = encrypt_image(img, enc_key)

    # sanity check: encryption/decryption reversibility
    decrypted_raw = decrypt_image(encrypted, enc_key)
    if not np.array_equal(decrypted_raw, img):
        diff = np.abs(decrypted_raw.astype(int) - img.astype(int))
        print(f"[CRIT] Encryption/Decryption mismatch on raw encrypted image {os.path.basename(image_path)}.")
        print(f"       Unique diffs sample: {np.unique(diff)[:10]}")

    # 4) Embed auxiliary bits into encrypted image
    t_embed_start = time.perf_counter()
    encrypted_with_aux, used_bits = embed_aux_into_encrypted(encrypted, label, aux_bits)
    t_embed_end = time.perf_counter()

    # 5) Decrypt the encrypted-with-aux image (used for extraction)
    decrypted_with_aux = decrypt_image(encrypted_with_aux, enc_key)



    # 6) Extract auxiliary bits
    t_extract_start = time.perf_counter()
    extracted_aux = extract_aux_from_encrypted(encrypted_with_aux, label, aux_len)
    t_extract_end = time.perf_counter()

    # Diagnostics
    capacity_bits = compute_capacity(label)
    print(f"[DIAG] {os.path.basename(image_path)}: aux_len={aux_len}, used_bits={used_bits}, capacity_bits={capacity_bits}")
    if aux_len > used_bits:
        print(f"[ERROR] aux_len ({aux_len}) > used_bits ({used_bits}) → not enough embedded bits.")
    elif aux_len > capacity_bits:
        print(f"[WARN] aux_len ({aux_len}) > capacity_bits ({capacity_bits}) → coded aux > theoretical capacity.")
    else:
        print(f"[DIAG] aux_len <= used_bits <= capacity OK.")
    print(f"[DIAG] aux_bits == extracted_aux ? {(aux_bits == extracted_aux)} (enc_len={len(aux_bits)}, ext_len={len(extracted_aux)})")

    # 7) Decode label map & recover image
    label_recovered = decode_label_map_from_bits(extracted_aux, rev, img.shape)

    # ✅ FIX 2: Use decrypted_raw (the one before embedding) for recovery.
    # This ensures perfect reversibility.
    recovered = recover_image_from_decrypted(decrypted_raw, label_recovered)

    # Label comparison
    try:
        label_mismatch = int(np.sum(label != label_recovered))
        print(f"[DIAG] Label mismatch count: {label_mismatch} / {h*w}")
    except Exception as e:
        print(f"[DIAG] Could not compare labels: {e}")

    # Final check
    exact = bool(np.array_equal(recovered, img))
    if exact:
        print(f"[OK] Recovered EXACT image for {os.path.basename(image_path)} ✅")
    else:
        diff = np.abs(recovered.astype(int) - img.astype(int))
        uniq = np.unique(diff)
        print(f"[FAIL] Recovered image differs. Unique diff sample: {uniq[:20]}")
        print(f"       min={diff.min()}, max={diff.max()}, mean={diff.mean():.4f}")
    # inside process_image_full_pipeline(), right after recovered computed and exact==False:
    # Detailed diagnostic for the first mismatch (replace your existing TRACE block)
    if not exact:
        # find first mismatch (skip row 0/col 0)
        mismatch = None
        for ii in range(1, h):
            for jj in range(1, w):
                if recovered[ii, jj] != img[ii, jj]:
                    mismatch = (ii, jj)
                    break
            if mismatch:
                break

        if mismatch:
            i, j = mismatch
            orig_px = int(img[i, j])
            rec_px = int(recovered[i, j])
            dec_px = int(decrypted_raw[i, j])
            # Label value decoded from aux
            lbl = int(label_recovered[i, j])
            # Recompute MED predictor at generation time using original image neighbors (a,b,c)
            # This replicates how label_map_generation should have computed pb when using MED.
            a = int(img[i, j-1])
            b = int(img[i-1, j])
            c = int(img[i-1, j-1])
            # MED formula used in label generation (simple a+b-c)
            p_gen = int(a + b - c)
            # But your label generation might use the MED variant with min/max handling (med_predictor)
            # so also compute the med-pixel predictor on original neighbors:
            if c <= min(a, b):
                p_med = max(a, b)
            elif c >= max(a, b):
                p_med = min(a, b)
            else:
                p_med = a + b - c

            # predictor used during recovery (reconstructed-neighbors predictor)
            p_rec = med_predictor_pixel(recovered.copy().astype(int), i, j)  # this reconstructs using current rec context

            # bit arrays (MSB-first)
            xb = ''.join(map(str, int_to_bits(orig_px, 8)))
            pb_gen = ''.join(map(str, int_to_bits(p_gen & 0xFF, 8)))
            pb_med = ''.join(map(str, int_to_bits(p_med & 0xFF, 8)))
            pb_rec = ''.join(map(str, int_to_bits(p_rec & 0xFF, 8)))
            dec_bits = ''.join(map(str, int_to_bits(dec_px, 8)))
            rec_bits = ''.join(map(str, int_to_bits(rec_px, 8)))

            # compute t_check = number of leading matching MSBs between orig and p_med (what label_map_generation should use)
            t_check = 0
            for k in range(8):
                if int(xb[k]) == int(pb_med[k]):
                    t_check += 1
                else:
                    break

            print(f"[TRACE-PIXEL] first mismatch at ({i},{j})")
            print(f"  orig={orig_px} {xb}")
            print(f"  decrypted_raw pixel={dec_px} {dec_bits}")
            print(f"  recovered={rec_px} {rec_bits}")
            print(f"  label_recovered (decoded) = {lbl}")
            print(f"  computed p_gen  (a+b-c) = {p_gen} {pb_gen}")
            print(f"  computed p_med  (med rule)= {p_med} {pb_med}")
            print(f"  predictor used at recovery (p_rec) = {p_rec} {pb_rec}")
            print(f"  t_check (matching MSBs orig vs p_med) = {t_check}")
            # quick sanity: what would label_map_generation produce if run locally for this pixel?
            print("  Notes:")
            if t_check == lbl:
                print("   -> label matches t_check (label consistent with MED predictor on original).")
            else:
                print("   -> label DOES NOT match t_check. This indicates label was produced using a different predictor or bit-order.")

    # if not exact:
    #     # find first mismatch
    #     mismatch = None
    #     for ii in range(1,h):
    #         for jj in range(1,w):
    #             if recovered[ii,jj] != img[ii,jj]:
    #                 mismatch = (ii,jj)
    #                 break
    #         if mismatch: break

    #     if mismatch:
    #         i,j = mismatch
    #         print(f"[TRACE-PIXEL] first mismatch at ({i},{j})")
    #         print("orig", img[i,j], bin(int(img[i,j]))[2:].zfill(8))
    #         # predictor used during recovery (reconstruct neighbor context up to this pixel)
    #         # We will compute px as recover does (replay recovery up to this pixel)
    #         tmp = decrypted_raw.copy().astype(int)
    #         for ii in range(0,i+1):
    #             for jj in range(0,w):
    #                 if ii==0 or jj==0:
    #                     tmp[ii,jj] = int(decrypted_raw[ii,jj])
    #                     continue
    #                 if ii==i and jj>j:
    #                     break
    #                 tt = int(label_recovered[ii,jj])
    #                 if tt==8:
    #                     tmp[ii,jj] = med_predictor_pixel(tmp, ii, jj)
    #                 else:
    #                     px_local = med_predictor_pixel(tmp, ii, jj)
    #                     p_bits_local = int_to_bits(px_local,8)
    #                     x_bits_local = int_to_bits(int(decrypted_raw[ii,jj]),8)
    #                     for k in range(0, tt):
    #                         x_bits_local[k] = p_bits_local[k]
    #                     x_bits_local[tt] = 1 - p_bits_local[tt]
    #                     # bits after t remain
    #                     tmp[ii,jj] = bits_to_int(x_bits_local)
    #             if ii==i: break

    #         # now compute px and bits for this pixel
    #         px = med_predictor_pixel(tmp, i, j)
    #         p_bits = int_to_bits(px,8)
    #         dec_bits = int_to_bits(int(decrypted_raw[i,j]),8)
    #         # compute reconstructed bits as our recover would
    #         x_bits = dec_bits.copy()
    #         for k in range(0, int(label_recovered[i,j])):
    #             x_bits[k] = p_bits[k]
    #         x_bits[int(label_recovered[i,j])] = 1 - p_bits[int(label_recovered[i,j])]
    #         print("predictor px", px, ''.join(map(str,p_bits)))
    #         print("decrypted_raw bits", ''.join(map(str,dec_bits)))
    #         print("reconstructed bits", ''.join(map(str,x_bits)), "->", bits_to_int(x_bits))
    #         print("recovered pixel", recovered[i,j], bin(int(recovered[i,j]))[2:].zfill(8))


    # Metrics
    mse, psnr = evaluate_arrays(img, recovered)
    metrics = {
        'height': h,
        'width': w,
        'aux_len': aux_len,
        'used_bits': used_bits,
        'capacity_bits': capacity_bits,
        'capacity_bpp': capacity_bits / float(h * w) if (h * w) > 0 else 0.0,
        'exact_reconstruction': int(exact),
        'mse': mse,
        'psnr_db': float('inf') if psnr == float('inf') else psnr,
        'embedding_time_s': (t_embed_end - t_embed_start),
        'extraction_time_s': (t_extract_end - t_extract_start),
        'time_total_s': (t_embed_end - t_embed_start) + (t_extract_end - t_extract_start)
    }
    return metrics



# ---------------- Dataset Processing ----------------
def hybrid_process(dataset_path, csv_path="metrics_hybrid_full.csv"):
    model_path = "cnn_model.h5"

    # Load or train CNN model
    model = None
    if os.path.exists(model_path):
        print("[INFO] Loading CNN model...")
        model = load_model(model_path, compile=False)
        model.make_predict_function()
    else:
        print("[INFO] Training CNN model...")
        model = train_cnn_model(dataset_path, model_path=model_path)

    # Collect sample images (first 10)
    image_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(dataset_path)
        for fname in files
        if fname.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]
    image_files = image_files[:10]
    total_images = len(image_files)
    print(f"[INFO] Found {total_images} valid image files.\n")

    rows = []
    skipped = []

    for idx, image_path in enumerate(sorted(image_files), start=1):
        rel_path = os.path.relpath(image_path, dataset_path)
        print(f"[INFO] ({idx}/{total_images}) Processing: {rel_path}")

        try:
            img = imageio.imread(image_path)
            if img is None or not isinstance(img, np.ndarray):
                print(f"[WARN] Skipping {rel_path}: invalid image data.")
                skipped.append(rel_path)
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(np.uint8)

            label_name, score = classify_image(image_path, threshold=1500)
            predictor_choice = 'CNN' if label_name == "complex" else 'MED'
            print(f"   ↳ {label_name.capitalize()} image detected (score={score:.2f}) → using {predictor_choice} predictor")

            start_full = time.perf_counter()
            metrics = process_image_full_pipeline(image_path, model=model, predictor=predictor_choice, enc_key='testkey')
            end_full = time.perf_counter()

            net_payload_bits = max(0, metrics["used_bits"] - metrics["aux_len"])
            net_payload_bpp = net_payload_bits / (metrics["height"] * metrics["width"])

            row = {
                "image_path": rel_path,
                "complexity_score": f"{score:.2f}",
                "label": label_name,
                "predictor": predictor_choice,
                "height": metrics["height"],
                "width": metrics["width"],
                "aux_len": metrics["aux_len"],
                "used_bits": metrics["used_bits"],
                "capacity_bits": metrics["capacity_bits"],
                "capacity_bpp": f"{metrics['capacity_bpp']:.6f}",
                "exact_reconstruction": metrics["exact_reconstruction"],
                "mse": f"{metrics['mse']:.10f}",
                "psnr_db": ("inf" if metrics["psnr_db"] == float("inf") else f"{metrics['psnr_db']:.6f}"),
                "embedding_time_s": f"{metrics['embedding_time_s']:.6f}",
                "extraction_time_s": f"{metrics['extraction_time_s']:.6f}",
                "time_total_s": f"{metrics['time_total_s']:.6f}",
                "full_run_elapsed_s": f"{(end_full - start_full):.6f}",
                "net_payload_bits": net_payload_bits,
                "net_payload_bpp": f"{net_payload_bpp:.6f}"
            }

            rows.append(row)

            if idx % 50 == 0:
                gc.collect()
                tf.keras.backend.clear_session()

        except Exception as e:
            print(f"[ERROR] Skipping {rel_path}: {e}")
            skipped.append(rel_path)
            continue

    # Save CSV
    if rows:
        fieldnames = [
            "image_path", "complexity_score", "label", "predictor",
            "height", "width", "aux_len", "used_bits", "capacity_bits",
            "capacity_bpp", "exact_reconstruction", "mse", "psnr_db",
            "embedding_time_s", "extraction_time_s", "time_total_s",
            "full_run_elapsed_s", "net_payload_bits", "net_payload_bpp"
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n✅ [INFO] Saved hybrid full-pipeline metrics to {os.path.abspath(csv_path)}")
        print(f"✅ [INFO] Total images processed successfully: {len(rows)} / {total_images}")

        if skipped:
            with open("skipped_images.txt", 'w') as f:
                for s in skipped:
                    f.write(s + "\n")
            print(f"⚠️ [INFO] Skipped {len(skipped)} images. Details saved in skipped_images.txt")
    else:
        print("[WARNING] No valid images found or processed.")


if __name__ == "__main__":
    dataset_path = "BOSSbase_1.01"
    hybrid_process(dataset_path)
