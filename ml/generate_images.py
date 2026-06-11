import os
import cv2
import numpy as np
import random
import math

# YOLOv12 Directory Mapping - Use absolute path
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "milk_evaporative_dataset")
SETS = ["train", "val"]
CLASSES = ["authentic", "water_diluted", "urea_added", "detergent_added"]

for s in SETS:
    os.makedirs(os.path.join(DATASET_DIR, "images", s), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", s), exist_ok=True)

def create_evaporative_base(size=640):
    """Generates a deep substrate canvas with a circular dried milk deposit area."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Substrate background noise (simulating microscopic slides/surfaces)
    bg_noise = np.random.normal(15, 3, img.shape).astype(np.uint8)
    img = cv2.add(img, bg_noise)

    center = (size // 2, size // 2)
    radius = random.randint(210, 245)

    # Isolate droplet boundary
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Render basic dried matrix foundation
    matrix = np.zeros_like(img)
    cv2.circle(matrix, center, radius, (45, 45, 45), -1)       # Central film
    cv2.circle(matrix, center, radius, (130, 130, 130), 5)     # Primary outer coffee-ring
    cv2.circle(matrix, center, radius-6, (80, 80, 80), 2)      # Secondary inner shadow ring

    return img, matrix, mask, center, radius

def inject_urea_dendrites(matrix, center, radius, complexity=16):
    """Generates branching crystalline structural lines characteristic of Urea evaporation."""
    for _ in range(complexity):
        angle = random.uniform(0, 2 * math.pi)
        start_radial_offset = random.uniform(0, radius * 0.25)

        # Crystal growth origin point
        cx = int(center[0] + start_radial_offset * math.cos(angle))
        cy = int(center[1] + start_radial_offset * math.sin(angle))

        growth_steps = random.randint(8, 12)
        segment_len = (radius * 0.85) / growth_steps

        for _ in range(growth_steps):
            # Advance crystal line with fractal variance
            next_x = int(cx + segment_len * math.cos(angle) + random.randint(-7, 7))
            next_y = int(cy + segment_len * math.sin(angle) + random.randint(-7, 7))

            # Draw main crystalline spine
            cv2.line(matrix, (cx, cy), (next_x, next_y), (225, 225, 225), random.randint(2, 4))

            # Secondary micro-needles branching sideways at 45/60 degrees
            if random.random() > 0.45:
                branch_angle = angle + random.choice([math.pi/4, -math.pi/4, math.pi/3])
                bx = int(next_x + random.randint(15, 30) * math.cos(branch_angle))
                by = int(next_y + random.randint(15, 30) * math.sin(branch_angle))
                cv2.line(matrix, (next_x, next_y), (bx, by), (185, 185, 185), 1)

            cx, cy = next_x, next_y

def inject_lipid_surfactant_pooling(matrix, center, radius, count=20):
    """Simulates ring-bound oil droplet merging and detergent film disruption."""
    for _ in range(count):
        # Surfactants migrate primarily toward intermediate or outer ring zones
        angle = random.uniform(0, 2 * math.pi)
        radial_distance = random.uniform(radius * 0.25, radius * 0.8)

        px = int(center[0] + radial_distance * math.cos(angle))
        py = int(center[1] + radial_distance * math.sin(angle))
        bubble_radius = random.randint(10, 32)

        # Concentric halo boundaries typical of fluid surfactant tension patterns
        cv2.circle(matrix, (px, py), bubble_radius, (145, 145, 145), 2)
        cv2.circle(matrix, (px, py), bubble_radius - 3, (65, 65, 65), -1)
        # Specular light refraction dots
        cv2.circle(matrix, (px - bubble_radius//3, py - bubble_radius//3), bubble_radius//6, (240, 240, 240), -1)

def build_dataset(volume_per_class=250, split=0.8):
    print(f"Initializing Dataset Construction aligned to DOI: 10.1021/acsomega.4c01274...")

    for class_id, class_name in enumerate(CLASSES):
        for idx in range(volume_per_class):
            bg, matrix, mask, center, radius = create_evaporative_base()

            if class_name == "authentic":
                # Clean natural sample: soft grain, homogeneous distribution
                grain = np.random.normal(0, 4, matrix.shape).astype(np.uint8)
                matrix = cv2.add(matrix, grain)

            elif class_name == "water_diluted":
                # Dilution limits evaporation density. Reduce opacity/contrast by 60%
                matrix = cv2.multiply(matrix, np.array([0.4, 0.4, 0.4, 1.0]))

            elif class_name == "urea_added":
                inject_urea_dendrites(matrix, center, radius, complexity=18)

            elif class_name == "detergent_added":
                inject_lipid_surfactant_pooling(matrix, center, radius, count=22)

            # Combine matrix deposit cleanly with our slide background via mask
            composited_frame = np.where(mask[:, :, None] == 255, matrix, bg)

            # Optical smoothing filter to emulate light microscope focus
            final_output = cv2.GaussianBlur(composited_frame, (3, 3), 0)

            # Derive Normalized YOLO Format Bounding Box [class x_center y_center width height]
            x_center = center[0] / 640.0
            y_center = center[1] / 640.0
            bbox_w = (radius * 2.0) / 640.0
            bbox_h = (radius * 2.0) / 640.0

            # Check Training vs Validation Allocation Split
            assignment = "train" if idx < (volume_per_class * split) else "val"
            file_identifier = f"{class_name}_{idx:04d}"

            # Write Image Binary and Label File
            cv2.imwrite(os.path.join(DATASET_DIR, "images", assignment, f"{file_identifier}.jpg"), final_output)
            with open(os.path.join(DATASET_DIR, "labels", assignment, f"{file_identifier}.txt"), "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

            print(f"[{class_id+1}/4] {class_name.upper()} - Generated: {file_identifier}.jpg | Set: {assignment}")

    print(f"Dataset successfully compiled inside parent directory: '{DATASET_DIR}'!")

if __name__ == "__main__":
    build_dataset(volume_per_class=250, split=0.8)