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


def process_image_full_pipeline(image_path, model=None, predictor='MED', enc_key='testkey',
                                enable_compression=False):
    """
    Run full MED/CNN image-level pipeline. If predictor=='CNN' we attempt CNN path;
    if the encoded auxiliary does not fit (aux_len > used_bits) or extraction fails,
    we automatically FALL BACK to MED predictor for that image (so reconstruction stays exact).
    """
    import os
    import zlib

    def bitstring_to_bytes(bs: str) -> bytes:
        if len(bs) == 0:
            return b''
        pad = (8 - (len(bs) % 8)) % 8
        bs_padded = bs + ('0' * pad)
        data = int(bs_padded, 2).to_bytes(len(bs_padded) // 8, byteorder='big')
        return bytes([pad]) + data

    def bytes_to_bitstring(b: bytes) -> str:
        if len(b) == 0:
            return ''
        pad = b[0]
        raw = b[1:]
        if len(raw) == 0:
            return ''
        bitlen = len(raw) * 8
        bs = bin(int.from_bytes(raw, 'big'))[2:].zfill(bitlen)
        if pad:
            bs = bs[:-pad]
        return bs

    # load image
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)
    h, w = img.shape

    # helper to run the core encode/encrypt/embed/extract/recover with a chosen predictor
    def try_run_with_predictor(use_predictor):
        # 1) label map generation
        label, counts = label_map_generation(img, predictor=use_predictor, cnn_model=model if use_predictor=='CNN' else None)

        # 2) mapping & encode
        mapping, rev = make_fixed_huffman_map(counts)
        aux_bits_uncompressed = encode_label_map(label, mapping)  # string of '0'/'1'

        # optional: compress aux bits to reduce aux_len
        if enable_compression:
            aux_bytes = bitstring_to_bytes(aux_bits_uncompressed)
            comp = zlib.compress(aux_bytes)
            aux_bits_to_embed = bytes_to_bitstring(comp)
            compressed = True
        else:
            aux_bits_to_embed = aux_bits_uncompressed
            compressed = False

        aux_len = len(aux_bits_to_embed)

        # 3) Encrypt (before embedding)
        encrypted, prng_mat = encrypt_image(img, enc_key)

        # 4) Embed aux bits into encrypted image
        t_embed_start = time.perf_counter()
        encrypted_with_aux, used_bits = embed_aux_into_encrypted(encrypted, label, aux_bits_to_embed)
        t_embed_end = time.perf_counter()

        # 5) Decrypt the encrypted-with-aux image (what receiver actually gets after decryption)
        decrypted_with_aux = decrypt_image(encrypted_with_aux, enc_key)

        # 6) Extract auxiliary bits (from encrypted_with_aux using label)
        t_extract_start = time.perf_counter()
        extracted_aux = extract_aux_from_encrypted(encrypted_with_aux, label, aux_len)
        t_extract_end = time.perf_counter()

        # If compressed, decompress to get original aux bitstring before decoding
        aux_bits_recovered = None
        try:
            if compressed:
                # convert extracted bitstring -> bytes -> decompress -> bitstring
                extracted_bytes = bitstring_to_bytes(extracted_aux)
                decomp = zlib.decompress(extracted_bytes)
                aux_bits_recovered = bytes_to_bitstring(decomp)
            else:
                aux_bits_recovered = extracted_aux
        except Exception as e:
            # decompression failed -> extraction corrupted or insufficient capacity
            aux_bits_recovered = None

        # decode if aux bits recovered ok
        label_recovered = None
        if aux_bits_recovered is not None:
            label_recovered = decode_label_map_from_bits(aux_bits_recovered, rev, img.shape)

        # For recovery we must use the decrypted image BEFORE embedding. In simulation we have it:
        decrypted_raw = decrypt_image(encrypted, enc_key)

        recovered = None
        if label_recovered is not None:
            # If predictor is CNN, we can use MED-based recovery only if labels were generated with MED.
            # But since we're doing image-level predictor, if use_predictor == 'CNN' we must ensure
            # recovery logic is consistent. For safety in this function we call recovery using MED
            # only when label map was created by MED. If CNN was used we still call recover_image_from_decrypted
            # because labels specify t-values; however recover_image_from_decrypted uses med_predictor_pixel.
            # This will fail if labels were produced using a different (CNN) predictor. So the caller should
            # check aux_bits and label_recovered consistency below and decide fallback.
            recovered = recover_image_from_decrypted(decrypted_raw, label_recovered)

        info = {
            'label': label,
            'counts': counts,
            'mapping': mapping,
            'rev': rev,
            'aux_bits_uncompressed': aux_bits_uncompressed,
            'aux_bits_to_embed': aux_bits_to_embed,
            'aux_len': aux_len,
            'used_bits': used_bits,
            'extracted_aux': extracted_aux,
            'aux_bits_recovered': aux_bits_recovered,
            'label_recovered': label_recovered,
            'decrypted_with_aux': decrypted_with_aux,
            'decrypted_raw': decrypted_raw,
            'encrypted_with_aux': encrypted_with_aux,
            't_embed': (t_embed_end - t_embed_start),
            't_extract': (t_extract_end - t_extract_start),
            'recovered': recovered
        }
        return info

    # ---------- Try chosen predictor path (CNN or MED) ----------
    run_info = try_run_with_predictor(predictor)

    # Diagnostics & fallback logic
    capacity_bits = compute_capacity(run_info['label'])
    print(f"[DIAG] {os.path.basename(image_path)}: aux_len={run_info['aux_len']}, used_bits={run_info['used_bits']}, capacity_bits={capacity_bits}")

    # condition 1: aux length cannot be more than used bits
    aux_fits = (run_info['aux_len'] <= run_info['used_bits'])
    # condition 2: we must be able to recover the original label bitstring (and decode to produce label_recovered)
    aux_recovered_ok = (run_info['aux_bits_recovered'] is not None and run_info['aux_bits_recovered'] == run_info['aux_bits_uncompressed'])
    # condition 3: label recovered should exist
    label_recovered_ok = (run_info['label_recovered'] is not None)

    if not aux_fits or not aux_recovered_ok or not label_recovered_ok:
        # fallback to MED predictor for exact reconstruction (this keeps behavior identical to base paper)
        print(f"[WARN] CNN path failed capacity/extraction or decoding for {os.path.basename(image_path)} — falling back to MED predictor for this image.")
        run_info = try_run_with_predictor('MED')  # rerun MED
        # recompute capacity with MED label
        capacity_bits = compute_capacity(run_info['label'])

        # After MED fallback, aux_bits_recovered should be fine (MED path proven)
        if run_info['label_recovered'] is None:
            raise RuntimeError("MED fallback unexpectedly failed to recover label map.")

    # Final check / diagnostics
    label_mismatch = 0
    try:
        label_mismatch = int(np.sum(run_info['label'] != run_info['label_recovered'])) if run_info['label_recovered'] is not None else -1
    except Exception:
        label_mismatch = -1

    # final recovered image
    recovered = run_info['recovered']
    decrypted_raw = run_info['decrypted_raw']

    # exact reconstruction?
    exact = bool(recovered is not None and np.array_equal(recovered, img))
    if exact:
        print(f"[OK] Recovered EXACT image for {os.path.basename(image_path)} ✅")
    else:
        # show some diagnostics
        if recovered is None:
            print(f"[FAIL] No recovered image for {os.path.basename(image_path)} (label_recovered missing).")
        else:
            diff = np.abs(recovered.astype(int) - img.astype(int))
            uniq = np.unique(diff)
            print(f"[FAIL] Recovered image differs. Unique diff sample: {uniq[:20]}")
            print(f"       min={diff.min()}, max={diff.max()}, mean={diff.mean():.4f}")

    # compute metrics
    mse, psnr = evaluate_arrays(img, recovered if recovered is not None else np.zeros_like(img))
    metrics = {
        'height': h,
        'width': w,
        'aux_len': run_info['aux_len'],
        'used_bits': run_info['used_bits'],
        'capacity_bits': capacity_bits,
        'capacity_bpp': capacity_bits / float(h * w) if (h * w) > 0 else 0.0,
        'exact_reconstruction': int(exact),
        'mse': mse,
        'psnr_db': float('inf') if psnr == float('inf') else psnr,
        'embedding_time_s': run_info['t_embed'],
        'extraction_time_s': run_info['t_extract'],
        'time_total_s': run_info['t_embed'] + run_info['t_extract']
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
    image_files = image_files[10:20]
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
