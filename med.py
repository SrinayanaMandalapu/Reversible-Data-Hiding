# med.py
"""
Reversible Data Hiding in Encrypted Images (fixed)
Converted and adapted for Colab from MATLAB implementation and previous draft.

Dependencies: numpy, opencv-python, imageio
Usage in Colab:
!pip install numpy opencv-python imageio
from rdhei_multi_msb_python_fixed import demo
res = demo('camera.png', enc_key='testkey')
"""
import numpy as np
import cv2
import imageio.v2 as imageio
from collections import Counter
from cnn import cnn_predictor  # import CNN predictor function
from utils import classify_image


# ----------------------- Utilities -----------------------

def int_to_bits(x, bits=8):
    """Return MSB-first bits list length `bits`."""
    return [ (x >> (bits-1-i)) & 1 for i in range(bits) ]

def bits_to_int(bits_list):
    """Convert MSB-first bits list back to integer."""
    v = 0
    for b in bits_list:
        v = (v << 1) | (int(b) & 1)
    return int(v)

# ----------------------- MED predictor -----------------------
# (used for label generation only; recovery uses per-pixel MED using reconstructed neighbors)

def med_predictor(img):
    """Compute MED predictor for each pixel (skip first row/col). Returns predicted image."""
    h, w = img.shape
    px = np.zeros_like(img, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0:
                px[i, j] = 0
            else:
                x1 = int(img[i, j-1])
                x2 = int(img[i-1, j])
                x3 = int(img[i-1, j-1])
                if x3 <= min(x1, x2):
                    px[i, j] = max(x1, x2)
                elif x3 >= max(x1, x2):
                    px[i, j] = min(x1, x2)
                else:
                    px[i, j] = x1 + x2 - x3
    return px

def med_predictor_pixel(rec, i, j):
    """MED predictor using already reconstructed neighbors for pixel (i,j).
    rec: 2D numpy array (int) that contains reconstructed pixels for already processed positions.
    For first row/col returns 0 (treated as reference)."""
    if i == 0 or j == 0:
        return 0
    x1 = int(rec[i, j-1])
    x2 = int(rec[i-1, j])
    x3 = int(rec[i-1, j-1])
    if x3 <= min(x1, x2):
        return max(x1, x2)
    elif x3 >= max(x1, x2):
        return min(x1, x2)
    else:
        return int(x1) + int(x2) - int(x3)

# ----------------------- Label map generation -----------------------

# def label_map_generation(img, predictor='MED', cnn_model=None):
#     """
#     Generate label map using either MED or CNN predictor.
#     """
#     h, w = img.shape
#     label = np.full((h, w), -1, dtype=np.int8)
#     counts = Counter()

#     for i in range(h):
#         for j in range(w):
#             if i == 0 or j == 0:
#                 counts[-1] += 1
#                 continue

#             # --- choose predictor ---
#             if predictor == 'CNN' and cnn_model is not None:
#                 p = cnn_predictor(img, i, j, cnn_model)
#             else:
#                 # MED predictor
#                 a = int(img[i, j-1])
#                 b = int(img[i-1, j])
#                 c = int(img[i-1, j-1])
#                 p = a + b - c  # MED formula

#             # label generation logic stays same
#             x = int(img[i, j])
#             xb = int_to_bits(x, 8)
#             pb = int_to_bits(p, 8)
#             t = 0
#             for k in range(8):
#                 if xb[k] == pb[k]:
#                     t += 1
#                 else:
#                     break
#             label[i, j] = t
#             counts[t] += 1

#     return label, counts

def label_map_generation(img, predictor='MED', cnn_model=None):
    """
    Generate label map using either MED or hybrid (MED + CNN) predictor.
    Ensures reversible predictions with integer outputs.
    """
    h, w = img.shape
    label = np.full((h, w), -1, dtype=np.int8)
    counts = Counter()

    if predictor == 'CNN':
        if cnn_model is None:
            raise ValueError("CNN predictor selected but no cnn_model provided")

        # --- Step 1: run CNN once to get predicted map ---
        from cnn import cnn_predictor
        predicted_map = cnn_predictor(img, cnn_model)  # shape same as img

        # Ensure pixel range [0,255] and type uint8
        if predicted_map.dtype != np.uint8:
            predicted_map = np.rint(predicted_map * 255).clip(0, 255).astype(np.uint8)

        # --- Step 2: compute MED predictor map ---
        from med import med_predictor
        med_map = med_predictor(img).astype(np.uint8)

        # --- Step 3: combine CNN + MED into a hybrid predictor (integer average) ---
        predicted_map = ((predicted_map.astype(np.int32) + med_map.astype(np.int32)) // 2).astype(np.uint8)

        # --- Step 4: generate label map using hybrid predictions ---
        for i in range(h):
            for j in range(w):
                if i == 0 or j == 0:
                    counts[-1] += 1
                    continue
                p = int(predicted_map[i, j])
                x = int(img[i, j])
                xb = int_to_bits(x, 8)
                pb = int_to_bits(p, 8)
                t = 0
                for k in range(8):
                    if xb[k] == pb[k]:
                        t += 1
                    else:
                        break
                label[i, j] = t
                counts[t] += 1

    else:
        # --- MED-only predictor path (unchanged) ---
        for i in range(h):
            for j in range(w):
                if i == 0 or j == 0:
                    counts[-1] += 1
                    continue

                a = int(img[i, j-1])      # left
                b = int(img[i-1, j])      # top
                c = int(img[i-1, j-1])    # top-left

                # --- True MED predictor (same as med_predictor_pixel) ---
                if c <= min(a, b):
                    p = max(a, b)
                elif c >= max(a, b):
                    p = min(a, b)
                else:
                    p = a + b - c

                # --- Bitwise match counting ---
                x = int(img[i, j])
                xb = int_to_bits(x, 8)
                pb = int_to_bits(p, 8)
                t = 0
                for k in range(8):
                    if xb[k] == pb[k]:
                        t += 1
                    else:
                        break
                label[i, j] = t
                counts[t] += 1


    return label, counts




# ----------------------- Huffman-like mapping -----------------------

def make_fixed_huffman_map(counts):
    """Use the paper's 9 fixed codewords and assign them to labels sorted by frequency descending."""
    codes_pool = ['00','01','100','101','1100','1101','1110','11110','11111']
    labels = [i for i in range(9)]
    label_counts = [(counts.get(l,0), l) for l in labels]
    label_counts.sort(reverse=True)
    mapping = {}
    for idx, (_, lbl) in enumerate(label_counts):
        mapping[lbl] = codes_pool[idx]
    rev = {v:k for k,v in mapping.items()}
    return mapping, rev

def encode_label_map(label, mapping):
    """Encode label map to a bitstring using mapping.
    Note: first row and column are reference pixels and are skipped in encoding
    (they must be stored/separately transmitted in a practical system).
    """
    h,w = label.shape
    bits = []
    for i in range(h):
        for j in range(w):
            if i==0 or j==0:
                # skip reference (paper stores parts of aux in these pixels separately)
                continue
            t = int(label[i,j])
            bits.extend(list(mapping[t]))
    return ''.join(bits)

# ----------------------- Encryption -----------------------

def generate_prng(enc_key, shape):
    """Deterministic pseudo-random generator based on key and image shape."""
    np.random.seed(sum(ord(c) for c in enc_key))  # basic key-based seed
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)

def encrypt_image(img, enc_key):
    """Reversible encryption using modular addition."""
    prng_mat = generate_prng(enc_key, img.shape)
    encrypted = (img.astype(np.uint16) + prng_mat.astype(np.uint16)) % 256
    return encrypted.astype(np.uint8), prng_mat

def decrypt_image(encrypted, enc_key, prng_mat=None):
    """Perfectly invertible decryption using modular subtraction."""
    if prng_mat is None:
        prng_mat = generate_prng(enc_key, encrypted.shape)
    decrypted = (encrypted.astype(np.uint16) - prng_mat.astype(np.uint16)) % 256
    return decrypted.astype(np.uint8)


# ----------------------- Embed auxiliary bits (multi-MSB) -----------------------

def embed_aux_into_encrypted(encrypted_img, label, aux_bits):
    """
    Multi-MSB embedding: for each pixel (skip first row/col),
    write (t+1) MSBs with the next aux bits (MSB-first).
    Returns modified encrypted image and number of bits written.
    """
    h, w = encrypted_img.shape
    out = encrypted_img.copy()
    bit_idx = 0
    total = len(aux_bits)
    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0:
                continue
            t = int(label[i, j])
            if t < 0:
                continue
            num_bits = (t + 1) if t <= 7 else 8
            # take next num_bits
            if bit_idx + num_bits <= total:
                to_write = aux_bits[bit_idx:bit_idx + num_bits]
                bit_idx += num_bits
            else:
                # pad remaining with zeros
                remaining = max(0, total - bit_idx)
                if remaining > 0:
                    to_write = aux_bits[bit_idx:bit_idx + remaining] + '0' * (num_bits - remaining)
                    bit_idx = total
                else:
                    to_write = '0' * num_bits

            pixel = int(out[i, j])
            # preserve lower (8-num_bits) bits
            lower_mask = (1 << (8 - num_bits)) - 1
            lower = pixel & lower_mask
            top_val = bits_to_int(list(to_write))  # to_write is MSB-first
            new_pixel = (top_val << (8 - num_bits)) | lower
            out[i, j] = np.uint8(new_pixel)
    return out, bit_idx


# ---------- extract auxiliary bits ----------
def extract_aux_from_encrypted(encrypted_img, label, aux_length_bits):
    """
    Read (t+1) MSBs from each pixel (skip first row/col) in the same order
    used during embedding. Return the extracted bitstring truncated to aux_length_bits.
    """
    h, w = encrypted_img.shape
    parts = []
    read = 0
    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0:
                continue
            t = int(label[i, j])
            if t < 0:
                continue
            num_bits = (t + 1) if t <= 7 else 8
            pixel = int(encrypted_img[i, j])
            # read top num_bits MSBs
            msb = (pixel >> (8 - num_bits)) & ((1 << num_bits) - 1)
            bits_part = bin(msb)[2:].zfill(num_bits)  # MSB-first string
            parts.append(bits_part)
            read += num_bits
            if read >= aux_length_bits:
                break
        if read >= aux_length_bits:
            break
    bitstring = ''.join(parts)
    return bitstring[:aux_length_bits]

# ----------------------- Decode label map (Huffman mapping known) -----------------------

def decode_label_map_from_bits(bitstring, rev_map, shape):
    """Decode the label map bitstring (which excludes first row/col) into a full label map array.
    rev_map: dict from code (string) -> label
    shape: (h,w)
    Note: this decoder assumes bitstring was produced by encode_label_map (same mapping).
    """
    h, w = shape
    label = np.zeros((h,w), dtype=np.int8)
    label[:, :] = -1
    i = 1       # START at row 1 (since row 0 is reference and was skipped when encoding)
    j = 1       # START at column 1 (since column 0 is reference)
    buffer = ""
    # iterate through bitstring and decode codewords using rev_map (prefix-free)
    for b in bitstring:
        buffer += b
        if buffer in rev_map:
            lbl = rev_map[buffer]
            if i >= h:
                # decoded more labels than expected — stop to avoid overflow
                break
            label[i, j] = lbl
            # advance to next pixel (skip column 0)
            j += 1
            if j == w:
                # end of row, wrap to next row (column 0 remains reference)
                j = 1
                i += 1
            buffer = ""
    # if buffer remains non-empty it may indicate truncated/extra bits; ignore for now
    return label

# ----------------------- Image recovery -----------------------

def recover_image_from_decrypted(decrypted_img, label):
    """
    Recover the original image from decrypted image and label map using MED predictor and Eq.(12).
    Uses MSB-first bit ordering.
    """
    h, w = decrypted_img.shape
    rec = decrypted_img.copy().astype(int)

    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0:
                rec[i, j] = int(decrypted_img[i, j])
                continue

            t = int(label[i, j])

            # Predictor using already reconstructed neighbors
            px = med_predictor_pixel(rec, i, j)

            if t == 8:
                rec[i, j] = int(px)
            else:
                # Convert to bits (MSB-first)
                p_bits = int_to_bits(px, 8)
                dec_bits = int_to_bits(int(decrypted_img[i, j]), 8)

                # Build the recovered pixel bits
                x_bits = p_bits.copy()

                # Flip the (t)-th bit
                x_bits[t] = 1 - p_bits[t]

                # Copy remaining bits (after t) from decrypted image
                for k in range(t+1, 8):
                    x_bits[k] = dec_bits[k]

                rec[i, j] = bits_to_int(x_bits)

    return np.clip(rec, 0, 255).astype(np.uint8)

def compute_capacity(label):
    """Compute total embedding capacity (bits) from label map."""
    h, w = label.shape
    cap = 0
    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0:  # skip reference pixels
                continue
            t = int(label[i, j])
            if t < 0:  # skip invalid
                continue
            cap += (t + 1) if t <= 7 else 8
    return cap

# ----------------------- Demo / Main -----------------------

def demo_med(image_path='lena.png', enc_key='enckey', show=False):
    """End-to-end demo:
     - loads grayscale image
     - generates label map & mapping
     - encodes label-map as aux bits (Huffman mapping)
     - encrypts image
     - embeds aux bits into encrypted image
     - decrypts and extracts aux bits (for simulation)
     - decodes label map and reconstructs original image
    Returns dict with arrays.
    """
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)
    print('Image loaded:', img.shape)

    # 1) Label map generation
    label, counts = label_map_generation(img)
    print('Label counts sample:', dict(list(counts.items())[:6]))

    # 2) Huffman mapping (paper's 9 codewords)
    mapping, rev = make_fixed_huffman_map(counts)
    print('Mapping (label -> code):', mapping)

    # 3) Encode label map to bitstring (excluding first row/col)
    aux_bits = encode_label_map(label, mapping)
    aux_len = len(aux_bits)
    print('Aux bits length:', aux_len)

    # 4) Encrypt
    encrypted, r = encrypt_image(img, enc_key)
    print('Encrypted image generated')

    # 5) Embed aux into encrypted image
    encrypted_with_aux, used = embed_aux_into_encrypted(encrypted, label, aux_bits)
    print(f'Embedded {used} aux bits into encrypted image')

    # --- At receiver: simulate extraction and recovery ---

    # 6) Decrypt (receiver has encryption key)
    decrypted = decrypt_image(encrypted_with_aux, enc_key)

    # 7) Extract auxiliary bits (simulate using known original label order)
    extracted_aux = extract_aux_from_encrypted(encrypted_with_aux, label, aux_len)
    # If this were realistic, mapping & aux_len would be stored in reference pixels.
    # 8) Decode label map from the extracted aux bits
    label_recovered = decode_label_map_from_bits(extracted_aux, rev, img.shape)

    # 9) Recover original image using recovered label
    recovered = recover_image_from_decrypted(decrypted, label_recovered)

    eq = np.array_equal(recovered, img)
    print('Reconstruction exact:', eq)
    if show:
        cv2.imshow('orig', img); cv2.imshow('rec', recovered); cv2.waitKey(0)

    label, counts = label_map_generation(img)
    capacity = compute_capacity(label)
    print(f"Embedding capacity: {capacity} bits")

    return {
        'original': img,
        'encrypted': encrypted_with_aux,
        'decrypted': decrypted,
        'recovered': recovered,
        'label': label,
        'label_recovered': label_recovered,
        'mapping': mapping,
        'aux_len': aux_len
    }