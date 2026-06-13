import os
import cv2
import numpy as np
import random
import math

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "milk_evaporative_dataset")
SETS = ["train", "val"]

CLASSES = [
    "authentic",
    "water_diluted",
    "urea_added",
    "ammonium_sulfate",
    "oil_surfactant",
    "formalin_added",
    "spoiled"
]

for s in SETS:
    os.makedirs(os.path.join(DATASET_DIR, "images", s), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", s), exist_ok=True)


def create_brightfield_base(size=640, brightness=235):
    """Creates a light grey/white brightfield microscopy background (matches reference images)."""
    img = np.full((size, size, 3), brightness, dtype=np.uint8)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # subtle vignette
    cx, cy = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size * 0.75)
    vignette = (1 - 0.08 * dist).clip(0, 1)
    img = (img.astype(np.float32) * vignette[:, :, None]).astype(np.uint8)

    center = (cx, cy)
    return img, center


def draw_fat_globule(img, cx, cy, radius, fill=255, edge=200):
    cv2.circle(img, (cx, cy), radius, (fill, fill, fill), -1)
    cv2.circle(img, (cx, cy), radius, (edge, edge, edge), 1)


def inject_authentic_globules(img, center, count=350):
    """Dense, evenly distributed small-to-medium white fat globules."""
    size = img.shape[0]
    for _ in range(count):
        x = random.randint(10, size - 10)
        y = random.randint(10, size - 10)
        r = random.randint(2, 9)
        draw_fat_globule(img, x, y, r, fill=random.randint(245, 255), edge=random.randint(195, 215))


def inject_water_diluted_globules(img, center, count=60):
    """Sparse small globules - simulates fat globule dilution."""
    size = img.shape[0]
    for _ in range(count):
        x = random.randint(10, size - 10)
        y = random.randint(10, size - 10)
        r = random.randint(2, 6)
        draw_fat_globule(img, x, y, r, fill=random.randint(245, 255), edge=random.randint(200, 220))


def inject_urea_dendrites(img, center, complexity=10):
    """Branching chain-like clusters of small spheres connecting into dendritic networks."""
    size = img.shape[0]
    cx0, cy0 = center
    for _ in range(complexity):
        x = random.randint(int(size * 0.1), int(size * 0.9))
        y = random.randint(int(size * 0.1), int(size * 0.9))
        angle = random.uniform(0, 2 * math.pi)
        steps = random.randint(10, 22)
        for step in range(steps):
            r = random.randint(3, 8)
            color = random.randint(225, 245)
            cv2.circle(img, (x, y), r, (color, color, color), -1)
            cv2.circle(img, (x, y), r, (200, 200, 200), 1)
            angle += random.uniform(-0.5, 0.5)
            step_len = random.randint(8, 16)
            x = int(np.clip(x + step_len * math.cos(angle), 0, size - 1))
            y = int(np.clip(y + step_len * math.sin(angle), 0, size - 1))
            # occasional small branch cluster
            if random.random() > 0.6:
                for _ in range(random.randint(2, 4)):
                    bx = int(np.clip(x + random.randint(-15, 15), 0, size - 1))
                    by = int(np.clip(y + random.randint(-15, 15), 0, size - 1))
                    br = random.randint(2, 5)
                    cv2.circle(img, (bx, by), br, (235, 235, 235), -1)


def inject_ams_nucleation(img, center, intensity=1.0):
    """Tight clusters of overlapping circular crystals plus thin radiating whiskers."""
    size = img.shape[0]
    num_clusters = int(8 * intensity)
    for _ in range(num_clusters):
        ccx = random.randint(int(size * 0.15), int(size * 0.85))
        ccy = random.randint(int(size * 0.15), int(size * 0.85))
        for _ in range(random.randint(8, 18)):
            x = ccx + random.randint(-30, 30)
            y = ccy + random.randint(-30, 30)
            r = random.randint(3, 14)
            cv2.circle(img, (x, y), r, (255, 255, 255), -1)
            cv2.circle(img, (x, y), r, (190, 190, 190), 1)
        # whiskers
        for _ in range(random.randint(3, 6)):
            angle = random.uniform(0, 2 * math.pi)
            length = random.randint(20, 60)
            ex = int(ccx + length * math.cos(angle))
            ey = int(ccy + length * math.sin(angle))
            cv2.line(img, (ccx, ccy), (ex, ey), (210, 210, 210), 1)


def inject_lipid_surfactant(img, center, count=8):
    """Large irregular dark amoeba-shaped blobs simulating chemical contamination/oil-surfactant films."""
    size = img.shape[0]
    for _ in range(count):
        ccx = random.randint(int(size * 0.1), int(size * 0.9))
        ccy = random.randint(int(size * 0.1), int(size * 0.9))
        base_r = random.randint(30, 90)
        pts = []
        n = random.randint(8, 14)
        for i in range(n):
            ang = 2 * math.pi * i / n
            rr = base_r * random.uniform(0.6, 1.2)
            px = int(ccx + rr * math.cos(ang))
            py = int(ccy + rr * math.sin(ang))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        shade = random.randint(60, 110)
        cv2.fillPoly(img, [pts], (shade, shade, shade))
        cv2.polylines(img, [pts], True, (30, 30, 30), 1)
        # small bubbles inside
        for _ in range(random.randint(3, 8)):
            bx = ccx + random.randint(-base_r // 2, base_r // 2)
            by = ccy + random.randint(-base_r // 2, base_r // 2)
            br = random.randint(2, 6)
            cv2.circle(img, (bx, by), br, (240, 240, 240), -1)


def inject_formalin_precipitation(img, center, roughness=1.0):
    """Large fused irregular grey plates with cracked, jagged outlines (protein fixation)."""
    size = img.shape[0]
    num_plates = int(6 * roughness)
    for _ in range(num_plates):
        ccx = random.randint(int(size * 0.1), int(size * 0.9))
        ccy = random.randint(int(size * 0.1), int(size * 0.9))
        base_r = random.randint(60, 160)
        n = random.randint(10, 18)
        pts = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            rr = base_r * random.uniform(0.5, 1.3)
            px = int(ccx + rr * math.cos(ang))
            py = int(ccy + rr * math.sin(ang))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        shade = random.randint(195, 225)
        cv2.fillPoly(img, [pts], (shade, shade, shade))
        cv2.polylines(img, [pts], True, (140, 140, 140), 2)
        # internal crack lines
        for _ in range(random.randint(3, 6)):
            x1 = ccx + random.randint(-base_r, base_r)
            y1 = ccy + random.randint(-base_r, base_r)
            x2 = x1 + random.randint(-40, 40)
            y2 = y1 + random.randint(-40, 40)
            cv2.line(img, (x1, y1), (x2, y2), (150, 150, 150), 1)


def inject_spoilage_degradation(img, center):
    """Scattered rod/curved bacterial-like dark marks plus diffuse dark clumps."""
    size = img.shape[0]
    # diffuse clumps
    for _ in range(10):
        ccx = random.randint(int(size * 0.1), int(size * 0.9))
        ccy = random.randint(int(size * 0.1), int(size * 0.9))
        r = random.randint(20, 60)
        shade = random.randint(150, 190)
        cv2.circle(img, (ccx, ccy), r, (shade, shade, shade), -1)
    # bacterial rod marks
    for _ in range(180):
        x = random.randint(5, size - 5)
        y = random.randint(5, size - 5)
        length = random.randint(4, 12)
        angle = random.uniform(0, 2 * math.pi)
        x2 = int(x + length * math.cos(angle))
        y2 = int(y + length * math.sin(angle))
        cv2.line(img, (x, y), (x2, y2), (20, 20, 20), 1)


# --- MASTER DATASET GENERATOR ---

def build_dataset(volume_per_class=150, mix_batch_volume=100, split=0.8):
    print("Initializing Multi-Class & Mixed Adulterant Dataset Engine...")

    for class_id, class_name in enumerate(CLASSES):
        for idx in range(volume_per_class):
            img, center = create_brightfield_base(size=640)

            if class_name == "authentic":
                inject_authentic_globules(img, center)
            elif class_name == "water_diluted":
                inject_water_diluted_globules(img, center)
            elif class_name == "urea_added":
                inject_authentic_globules(img, center, count=150)
                inject_urea_dendrites(img, center)
            elif class_name == "ammonium_sulfate":
                inject_authentic_globules(img, center, count=150)
                inject_ams_nucleation(img, center)
            elif class_name == "oil_surfactant":
                inject_authentic_globules(img, center, count=200)
                inject_lipid_surfactant(img, center)
            elif class_name == "formalin_added":
                inject_authentic_globules(img, center, count=120)
                inject_formalin_precipitation(img, center)
            elif class_name == "spoiled":
                inject_authentic_globules(img, center, count=100)
                inject_spoilage_degradation(img, center)

            save_image_and_label(img, [class_id], class_name, idx, volume_per_class, split)

    # Phase 2: Mixed batches
    mixtures = [
        ("diluted_urea_ams", [CLASSES.index("water_diluted"), CLASSES.index("urea_added"), CLASSES.index("ammonium_sulfate")]),
        ("urea_oil_surfactant", [CLASSES.index("urea_added"), CLASSES.index("oil_surfactant")]),
        ("diluted_oil_formalin", [CLASSES.index("water_diluted"), CLASSES.index("oil_surfactant"), CLASSES.index("formalin_added")]),
        ("all_inclusive_fraud", [CLASSES.index("water_diluted"), CLASSES.index("urea_added"), CLASSES.index("ammonium_sulfate"), CLASSES.index("oil_surfactant")])
    ]

    for mix_name, active_ids in mixtures:
        for idx in range(mix_batch_volume):
            img, center = create_brightfield_base(size=640)

            base_count = 60 if CLASSES.index("water_diluted") in active_ids else 150
            inject_authentic_globules(img, center, count=base_count)

            if CLASSES.index("urea_added") in active_ids:
                inject_urea_dendrites(img, center, complexity=7)
            if CLASSES.index("ammonium_sulfate") in active_ids:
                inject_ams_nucleation(img, center, intensity=0.7)
            if CLASSES.index("oil_surfactant") in active_ids:
                inject_lipid_surfactant(img, center, count=5)
            if CLASSES.index("formalin_added") in active_ids:
                inject_formalin_precipitation(img, center, roughness=0.6)

            save_image_and_label(img, active_ids, mix_name, idx, mix_batch_volume, split)

    print(f"\nDataset successfully compiled inside parent directory: '{DATASET_DIR}'!")


def save_image_and_label(img, class_ids, identifier, idx, total_vol, split):
    """Applies final blur/noise and exports YOLO classification-style dataset assets
    (single full-frame bounding box, since the whole frame is the sample)."""
    final_output = cv2.GaussianBlur(img, (3, 3), 0.6)
    noise = np.random.normal(0, 1.5, final_output.shape).astype(np.int16)
    final_output = np.clip(final_output.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Full-frame bounding box (entire image is the labeled region)
    x_center, y_center, bbox_w, bbox_h = 0.5, 0.5, 1.0, 1.0

    assignment = "train" if idx < (total_vol * split) else "val"
    file_id = f"{identifier}_{idx:04d}"

    cv2.imwrite(os.path.join(DATASET_DIR, "images", assignment, f"{file_id}.jpg"), final_output)

    with open(os.path.join(DATASET_DIR, "labels", assignment, f"{file_id}.txt"), "w") as f:
        for cid in class_ids:
            f.write(f"{cid} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")


if __name__ == "__main__":
    build_dataset(volume_per_class=150, mix_batch_volume=100, split=0.8)