# gap.py (formerly med.py)
import numpy as np
import cv2
import imageio.v2 as imageio
from collections import Counter
import math, time

# ----------------------- Utilities -----------------------
def int_to_bits(x, bits=8):
    return [(x >> (bits-1-i)) & 1 for i in range(bits)]

def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | (int(b) & 1)
    return int(v)

# ----------------------- GAP predictor -----------------------
def gap_predictor(img, threshold=8):
    h, w = img.shape
    px = np.zeros_like(img, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0:
                px[i, j] = 0
            else:
                a = int(img[i, j-1])     # left
                b = int(img[i-1, j])     # top
                c = int(img[i-1, j-1])   # top-left
                dh = abs(a - c)
                dv = abs(b - c)
                if dh - dv > threshold:
                    p = a
                elif dv - dh > threshold:
                    p = b
                else:
                    p = (a + b) // 2
                px[i, j] = p
    return px

def gap_predictor_pixel(rec, i, j, threshold=8):
    if i == 0 or j == 0:
        return 0
    a = int(rec[i, j-1])
    b = int(rec[i-1, j])
    c = int(rec[i-1, j-1])
    dh = abs(a - c)
    dv = abs(b - c)
    if dh - dv > threshold:
        return a
    elif dv - dh > threshold:
        return b
    else:
        return (a + b) // 2

# ----------------------- Label map -----------------------
def label_map_generation(img):
    h, w = img.shape
    px = gap_predictor(img)
    label = np.full((h, w), -1, dtype=np.int8)
    counts = Counter()
    for i in range(h):
        for j in range(w):
            if i==0 or j==0:
                counts[-1] += 1
                continue
            x = int(img[i,j])
            p = int(px[i,j])
            xb = int_to_bits(x,8)
            pb = int_to_bits(p,8)
            t = 0
            for k in range(8):
                if xb[k] == pb[k]:
                    t += 1
                else:
                    break
            label[i,j] = t
            counts[int(t)] += 1
    return label, counts

# ----------------------- Huffman-like mapping -----------------------
def make_fixed_huffman_map(counts):
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
    h,w = label.shape
    bits = []
    for i in range(h):
        for j in range(w):
            if i==0 or j==0:
                continue
            t = int(label[i,j])
            bits.extend(list(mapping[t]))
    return ''.join(bits)

# ----------------------- Encryption -----------------------
def generate_prng_matrix(shape, key):
    rng = np.random.RandomState(abs(hash(key)) % (2**32))
    return rng.randint(0,256,size=shape,dtype=np.uint8)

def encrypt_image(img, key):
    r = generate_prng_matrix(img.shape, key)
    return np.bitwise_xor(img, r), r

def decrypt_image(encrypted_img, key):
    r = generate_prng_matrix(encrypted_img.shape, key)
    return np.bitwise_xor(encrypted_img, r)

# ----------------------- Embed / Extract -----------------------
def embed_aux_into_encrypted(encrypted_img, label, aux_bits):
    h,w = encrypted_img.shape
    out = encrypted_img.copy()
    bit_idx = 0
    total = len(aux_bits)
    for i in range(h):
        for j in range(w):
            if i==0 or j==0:
                continue
            t = int(label[i,j])
            if t < 0:
                continue
            num_bits = (t+1) if t <= 7 else 8
            if bit_idx + num_bits <= total:
                to_write = aux_bits[bit_idx:bit_idx+num_bits]
                bit_idx += num_bits
            else:
                remaining = max(0, total - bit_idx)
                if remaining > 0:
                    to_write = aux_bits[bit_idx:bit_idx+remaining] + '0'*(num_bits-remaining)
                    bit_idx = total
                else:
                    to_write = '0'*num_bits
            pixel = int(out[i,j])
            lower_mask = (1 << (8-num_bits)) - 1
            lower = pixel & lower_mask
            new_pixel = (bits_to_int(list(to_write)) << (8-num_bits)) | lower
            out[i,j] = np.uint8(new_pixel)
    return out, bit_idx

def extract_aux_from_encrypted(encrypted_img, label, aux_length_bits):
    h,w = encrypted_img.shape
    read = 0
    parts = []
    for i in range(h):
        for j in range(w):
            if i==0 or j==0:
                continue
            t = int(label[i,j])
            if t < 0:
                continue
            num_bits = (t+1) if t <= 7 else 8
            pixel = int(encrypted_img[i,j])
            msb = (pixel >> (8-num_bits)) & ((1<<num_bits)-1)
            parts.append(bin(msb)[2:].zfill(num_bits))
            read += num_bits
            if read >= aux_length_bits:
                break
        if read >= aux_length_bits:
            break
    bitstring = ''.join(parts)
    return bitstring[:aux_length_bits]

# ----------------------- Decode label bits -----------------------
def decode_label_map_from_bits(bitstring, rev_map, shape):
    h, w = shape
    label = np.full((h,w), -1, dtype=np.int8)
    i, j = 1, 1
    buffer = ""
    for b in bitstring:
        buffer += b
        if buffer in rev_map:
            lbl = rev_map[buffer]
            if i >= h:
                break
            label[i, j] = lbl
            j += 1
            if j == w:
                j = 1
                i += 1
            buffer = ""
    return label

# ----------------------- Image recovery -----------------------
def recover_image_from_decrypted(decrypted_img, label):
    h,w = decrypted_img.shape
    rec = decrypted_img.copy().astype(int)
    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0:
                rec[i,j] = int(decrypted_img[i,j])
                continue
            t = int(label[i,j])
            px = gap_predictor_pixel(rec, i, j)
            if t == 8:
                rec[i,j] = int(px)
            else:
                p_bits = int_to_bits(px, 8)
                x_bits = p_bits.copy()
                x_bits[t] = 1 - p_bits[t]
                pixel_decrypted = int(decrypted_img[i, j])
                dec_bits = int_to_bits(pixel_decrypted, 8)
                for k in range(t+1, 8):
                    x_bits[k] = dec_bits[k]
                rec[i, j] = bits_to_int(x_bits)
    return np.clip(rec, 0, 255).astype(np.uint8)

# ----------------------- Capacity -----------------------
def compute_capacity_np(label: np.ndarray) -> int:
    core = label.copy()
    core[0, :] = -1
    core[:, 0] = -1
    valid = core >= 0
    bits = np.where(core <= 7, core + 1, 8)
    return int(bits[valid].sum())

# ----------------------- Demo -----------------------
def demo_gap(image_path='lena.png', enc_key='enckey', show=False):
    t0 = time.perf_counter()
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)
    h, w = img.shape

    label, counts = label_map_generation(img)
    mapping, rev = make_fixed_huffman_map(counts)
    aux_bits = encode_label_map(label, mapping)
    aux_len = len(aux_bits)

    encrypted, _ = encrypt_image(img, enc_key)
    encrypted_with_aux, used = embed_aux_into_encrypted(encrypted, label, aux_bits)
    decrypted = decrypt_image(encrypted_with_aux, enc_key)
    extracted_aux = extract_aux_from_encrypted(encrypted_with_aux, label, aux_len)
    label_recovered = decode_label_map_from_bits(extracted_aux, rev, img.shape)
    recovered = recover_image_from_decrypted(decrypted, label_recovered)

    exact = bool(np.array_equal(recovered, img))
    diff = img.astype(np.float64) - recovered.astype(np.float64)
    mse = float(np.mean(diff * diff))
    psnr = float('inf') if mse == 0.0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    capacity_bits = compute_capacity_np(label)
    capacity_bpp = capacity_bits / float(h * w)
    total_time_s = time.perf_counter() - t0

    return {
        'original': img,
        'recovered': recovered,
        'height': h,
        'width': w,
        'aux_len': aux_len,
        'used_bits': used,
        'capacity_bits': capacity_bits,
        'capacity_bpp': capacity_bpp,
        'exact_reconstruction': exact,
        'mse': mse,
        'psnr_db': psnr,
        'time_total_s': total_time_s
    }
