import os
import cv2
import numpy as np
import random
import math

# YOLOv12 Directory Mapping - Use absolute path
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "milk_evaporative_dataset")
SETS = ["train", "val"]
CLASSES = ["authentic", "water_diluted", "urea_added", "detergent_added", "formalin_added", "h2o2_added", "spoiled"]

for s in SETS:
    os.makedirs(os.path.join(DATASET_DIR, "images", s), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", s), exist_ok=True)

def create_evaporative_base(size=240):
    """Generates a deep substrate canvas with a circular dried milk deposit area with realistic milk colors."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Substrate background noise with light brownish tint (simulating microscopic slides/surfaces)
    bg_noise = np.random.normal(20, 4, img.shape).astype(np.uint8)
    img[:,:,0] = np.clip(img[:,:,0] + bg_noise[:,:,0] + 10, 0, 255)  # Blue channel
    img[:,:,1] = np.clip(img[:,:,1] + bg_noise[:,:,1] + 15, 0, 255)  # Green channel
    img[:,:,2] = np.clip(img[:,:,2] + bg_noise[:,:,2] + 20, 0, 255)  # Red channel (warmer tone)

    center = (size // 2, size // 2)
    radius = random.randint(210, 245)

    # Isolate droplet boundary
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Render basic dried milk matrix foundation with realistic milk-brown coloration
    matrix = np.zeros_like(img)
    # Central dried milk deposit - warm milk brown
    cv2.circle(matrix, center, radius, (70, 90, 110), -1)
    # Primary outer coffee-ring - darker brown
    cv2.circle(matrix, center, radius, (100, 130, 160), 8)
    # Secondary inner shadow ring - deep brown
    cv2.circle(matrix, center, radius-8, (60, 80, 100), 3)
    # Additional subtle rings for milk sediment layering
    cv2.circle(matrix, center, int(radius*0.85), (80, 105, 130), 2)

    return img, matrix, mask, center, radius

def inject_urea_dendrites(matrix, center, radius, complexity=16):
    """Generates branching crystalline structural lines characteristic of Urea evaporation with enhanced detail."""
    for _ in range(complexity):
        angle = random.uniform(0, 2 * math.pi)
        start_radial_offset = random.uniform(0, radius * 0.3)

        # Crystal growth origin point
        cx = int(center[0] + start_radial_offset * math.cos(angle))
        cy = int(center[1] + start_radial_offset * math.sin(angle))

        growth_steps = random.randint(10, 16)
        segment_len = (radius * 0.85) / growth_steps

        for step in range(growth_steps):
            # Advance crystal line with fractal variance
            next_x = int(cx + segment_len * math.cos(angle) + random.randint(-8, 8))
            next_y = int(cy + segment_len * math.sin(angle) + random.randint(-8, 8))

            # Draw main crystalline spine with milk-brown base color
            thickness = max(1, random.randint(2, 5) - (step // 4))
            spine_color = (180 + random.randint(-20, 20), 200 + random.randint(-20, 20), 220 + random.randint(-20, 20))
            cv2.line(matrix, (cx, cy), (next_x, next_y), spine_color, thickness)

            # Secondary micro-needles branching at multiple angles
            if random.random() > 0.40:
                for branch_offset in [math.pi/4, -math.pi/4, math.pi/3, -math.pi/3]:
                    if random.random() > 0.65:
                        branch_angle = angle + branch_offset
                        branch_length = random.randint(12, 35)
                        bx = int(next_x + branch_length * math.cos(branch_angle))
                        by = int(next_y + branch_length * math.sin(branch_angle))
                        branch_color = (160 + random.randint(-15, 15), 180 + random.randint(-15, 15), 200 + random.randint(-15, 15))
                        cv2.line(matrix, (next_x, next_y), (bx, by), branch_color, max(1, thickness - 1))

            # Fine crystalline texture spots
            if random.random() > 0.75:
                spot_radius = random.randint(1, 3)
                spot_color = (200 + random.randint(-10, 10), 210 + random.randint(-10, 10), 230 + random.randint(-10, 10))
                cv2.circle(matrix, (next_x, next_y), spot_radius, spot_color, -1)

            cx, cy = next_x, next_y

def inject_lipid_surfactant_pooling(matrix, center, radius, count=20):
    """Simulates ring-bound oil droplet merging and detergent film disruption with enhanced pooling effects."""
    for _ in range(count):
        # Surfactants migrate primarily toward intermediate or outer ring zones
        angle = random.uniform(0, 2 * math.pi)
        radial_distance = random.uniform(radius * 0.25, radius * 0.85)

        px = int(center[0] + radial_distance * math.cos(angle))
        py = int(center[1] + radial_distance * math.sin(angle))
        bubble_radius = random.randint(12, 40)

        # Primary bubble with milk-brown gradient
        outer_color = (120 + random.randint(-20, 20), 140 + random.randint(-20, 20), 160 + random.randint(-20, 20))
        inner_color = (90 + random.randint(-15, 15), 110 + random.randint(-15, 15), 130 + random.randint(-15, 15))

        # Concentric halo boundaries typical of fluid surfactant tension patterns
        cv2.circle(matrix, (px, py), bubble_radius, outer_color, 3)
        cv2.circle(matrix, (px, py), bubble_radius - 4, inner_color, -1)

        # Enhanced specular light refraction with multiple highlights
        highlight_offset = bubble_radius // 3
        cv2.circle(matrix, (px - highlight_offset, py - highlight_offset), max(1, bubble_radius//5), (240, 245, 250), -1)

        # Secondary smaller highlights for optical effect
        if random.random() > 0.6:
            cv2.circle(matrix, (px + highlight_offset//2, py - highlight_offset//2), max(1, bubble_radius//8), (230, 235, 245), -1)

        # Micro-droplet clustering around main bubble
        for _ in range(random.randint(2, 5)):
            micro_angle = random.uniform(0, 2 * math.pi)
            micro_dist = random.randint(bubble_radius + 5, bubble_radius + 20)
            micro_x = int(px + micro_dist * math.cos(micro_angle))
            micro_y = int(py + micro_dist * math.sin(micro_angle))
            micro_radius = random.randint(3, 8)
            micro_color = (140 + random.randint(-15, 15), 160 + random.randint(-15, 15), 180 + random.randint(-15, 15))
            cv2.circle(matrix, (micro_x, micro_y), micro_radius, micro_color, -1)

def build_dataset(volume_per_class=250, split=0.8):
    print(f"Initializing Dataset Construction aligned to DOI: 10.1021/acsomega.4c01274...")

    for class_id, class_name in enumerate(CLASSES):
        for idx in range(volume_per_class):
            bg, matrix, mask, center, radius = create_evaporative_base()

            if class_name == "authentic":
                # Clean natural sample: fine granular texture, homogeneous distribution
                # Add natural milk protein grain texture
                grain = np.random.normal(0, 3, matrix.shape).astype(np.int16)
                matrix = np.clip(matrix.astype(np.int16) + grain, 0, 255).astype(np.uint8)

                # Add very subtle natural variation in milk deposit
                for i in range(3):
                    small_circle_x = center[0] + random.randint(-int(radius*0.3), int(radius*0.3))
                    small_circle_y = center[1] + random.randint(-int(radius*0.3), int(radius*0.3))
                    small_radius = random.randint(20, 50)
                    overlay_color = (75 + random.randint(-10, 10), 95 + random.randint(-10, 10), 115 + random.randint(-10, 10))
                    cv2.circle(matrix, (small_circle_x, small_circle_y), small_radius, overlay_color, random.randint(1, 2))

            elif class_name == "water_diluted":
                # Dilution limits evaporation density. Reduce contrast and create washed-out appearance
                # Water dilution causes lighter, less concentrated deposits
                matrix = cv2.multiply(matrix, np.array([0.55, 0.55, 0.55, 1.0]))

                # Add fluid flow patterns (water tends to create smoother, radial patterns)
                for i in range(8):
                    flow_angle = (2 * math.pi * i) / 8
                    start_x = int(center[0] + (radius * 0.3) * math.cos(flow_angle))
                    start_y = int(center[1] + (radius * 0.3) * math.sin(flow_angle))
                    end_x = int(center[0] + (radius * 0.85) * math.cos(flow_angle))
                    end_y = int(center[1] + (radius * 0.85) * math.sin(flow_angle))
                    flow_color = (100 + random.randint(-10, 10), 120 + random.randint(-10, 10), 140 + random.randint(-10, 10))
                    cv2.line(matrix, (start_x, start_y), (end_x, end_y), flow_color, random.randint(1, 2))

            elif class_name == "urea_added":
                # Urea creates distinct crystalline dendrite patterns
                inject_urea_dendrites(matrix, center, radius, complexity=20)

                # Add secondary fine crystalline texture overlay
                for _ in range(random.randint(100, 150)):
                    spot_x = random.randint(center[0] - radius, center[0] + radius)
                    spot_y = random.randint(center[1] - radius, center[1] + radius)
                    if (spot_x - center[0])**2 + (spot_y - center[1])**2 <= radius**2:
                        spot_size = random.randint(1, 2)
                        spot_color = (200 + random.randint(-15, 15), 215 + random.randint(-15, 15), 230 + random.randint(-15, 15))
                        cv2.circle(matrix, (spot_x, spot_y), spot_size, spot_color, -1)

            elif class_name == "detergent_added":
                # Detergent creates distinct pooling, bubble formation, and disruption patterns
                inject_lipid_surfactant_pooling(matrix, center, radius, count=26)

                # Add disruption zones - areas where detergent breaks surface tension
                for _ in range(random.randint(12, 18)):
                    disrupt_x = random.randint(center[0] - radius, center[0] + radius)
                    disrupt_y = random.randint(center[1] - radius, center[1] + radius)
                    if (disrupt_x - center[0])**2 + (disrupt_y - center[1])**2 <= radius**2:
                        disrupt_radius = random.randint(15, 35)
                        # Lighter disruption zone
                        cv2.circle(matrix, (disrupt_x, disrupt_y), disrupt_radius, (150, 170, 190), 1)
                        # Fill with lighter color
                        if random.random() > 0.5:
                            cv2.circle(matrix, (disrupt_x, disrupt_y), max(1, disrupt_radius - 3), (140, 160, 180), -1)

            elif class_name == "formalin_added":
                # Formalin creates harsh, dark, densely precipitated deposits with rough texture
                # Darkens the deposit significantly
                matrix = cv2.multiply(matrix, np.array([0.7, 0.75, 0.8, 1.0]))

                # Create harsh, chunky texture with rough edges
                for _ in range(random.randint(80, 120)):
                    rough_x = random.randint(center[0] - radius, center[0] + radius)
                    rough_y = random.randint(center[1] - radius, center[1] + radius)
                    if (rough_x - center[0])**2 + (rough_y - center[1])**2 <= radius**2:
                        rough_radius = random.randint(3, 12)
                        rough_color = (50 + random.randint(-15, 15), 60 + random.randint(-15, 15), 80 + random.randint(-15, 15))
                        cv2.circle(matrix, (rough_x, rough_y), rough_radius, rough_color, -1)

                # Add irregular linear striations (harsh chemical precipitation)
                for _ in range(random.randint(10, 15)):
                    stripe_angle = random.uniform(0, 2 * math.pi)
                    stripe_start_x = int(center[0] + (radius * 0.4) * math.cos(stripe_angle))
                    stripe_start_y = int(center[1] + (radius * 0.4) * math.sin(stripe_angle))
                    stripe_end_x = int(center[0] + (radius * 0.95) * math.cos(stripe_angle))
                    stripe_end_y = int(center[1] + (radius * 0.95) * math.sin(stripe_angle))
                    stripe_color = (60 + random.randint(-10, 10), 70 + random.randint(-10, 10), 90 + random.randint(-10, 10))
                    cv2.line(matrix, (stripe_start_x, stripe_start_y), (stripe_end_x, stripe_end_y), stripe_color, random.randint(2, 4))

            elif class_name == "h2o2_added":
                # H2O2 creates oxidative patterns - lighter, more bleached appearance with radial streaking
                # Hydrogen peroxide bleaches deposits, creating lighter zones
                matrix = cv2.multiply(matrix, np.array([0.65, 0.7, 0.75, 1.0]))

                # Add radial oxidation streaks - lighter lines emanating from center
                for i in range(12):
                    oxidation_angle = (2 * math.pi * i) / 12 + random.uniform(-0.2, 0.2)
                    ox_start_x = int(center[0] + (radius * 0.2) * math.cos(oxidation_angle))
                    ox_start_y = int(center[1] + (radius * 0.2) * math.sin(oxidation_angle))
                    ox_end_x = int(center[0] + (radius * 0.9) * math.cos(oxidation_angle))
                    ox_end_y = int(center[1] + (radius * 0.9) * math.sin(oxidation_angle))
                    ox_color = (140 + random.randint(-10, 15), 160 + random.randint(-10, 15), 180 + random.randint(-10, 15))
                    cv2.line(matrix, (ox_start_x, ox_start_y), (ox_end_x, ox_end_y), ox_color, random.randint(1, 3))

                # Add bleached spots - lighter circular zones
                for _ in range(random.randint(15, 25)):
                    bleach_x = random.randint(center[0] - radius, center[0] + radius)
                    bleach_y = random.randint(center[1] - radius, center[1] + radius)
                    if (bleach_x - center[0])**2 + (bleach_y - center[1])**2 <= radius**2:
                        bleach_radius = random.randint(8, 20)
                        bleach_color = (150 + random.randint(-10, 20), 170 + random.randint(-10, 20), 190 + random.randint(-10, 20))
                        cv2.circle(matrix, (bleach_x, bleach_y), bleach_radius, bleach_color, -1)

            elif class_name == "spoiled":
                # Spoilage creates very dark, dense deposits with irregular degradation patterns
                # Creates dark coloration indicating bacterial degradation
                matrix = cv2.multiply(matrix, np.array([0.5, 0.55, 0.6, 1.0]))

                # Add dark degradation particles
                for _ in range(random.randint(120, 180)):
                    degrade_x = random.randint(center[0] - radius, center[0] + radius)
                    degrade_y = random.randint(center[1] - radius, center[1] + radius)
                    if (degrade_x - center[0])**2 + (degrade_y - center[1])**2 <= radius**2:
                        degrade_radius = random.randint(2, 8)
                        degrade_color = (30 + random.randint(-10, 20), 40 + random.randint(-10, 20), 60 + random.randint(-10, 20))
                        cv2.circle(matrix, (degrade_x, degrade_y), degrade_radius, degrade_color, -1)

                # Add irregular dark clumps (bacterial colonies)
                for _ in range(random.randint(8, 15)):
                    clump_center_x = random.randint(center[0] - radius, center[0] + radius)
                    clump_center_y = random.randint(center[1] - radius, center[1] + radius)
                    if (clump_center_x - center[0])**2 + (clump_center_y - center[1])**2 <= radius**2:
                        clump_base_radius = random.randint(20, 50)
                        # Rough, irregular clump boundaries
                        clump_color = (35 + random.randint(-15, 10), 45 + random.randint(-15, 10), 65 + random.randint(-15, 10))
                        cv2.circle(matrix, (clump_center_x, clump_center_y), clump_base_radius, clump_color, -1)
                        # Add sub-clumps for irregular texture
                        for _ in range(random.randint(3, 6)):
                            sub_offset_x = random.randint(-20, 20)
                            sub_offset_y = random.randint(-20, 20)
                            sub_radius = random.randint(5, 15)
                            sub_color = (25 + random.randint(-10, 15), 35 + random.randint(-10, 15), 55 + random.randint(-10, 15))
                            cv2.circle(matrix, (clump_center_x + sub_offset_x, clump_center_y + sub_offset_y), sub_radius, sub_color, -1)

            # Combine matrix deposit cleanly with our slide background via mask
            composited_frame = np.where(mask[:, :, None] == 255, matrix, bg)

            # Optical smoothing filter to emulate light microscope focus
            final_output = cv2.GaussianBlur(composited_frame, (5, 5), 1.2)

            # Add subtle noise to simulate optical grain
            noise = np.random.normal(0, 2, final_output.shape).astype(np.int16)
            final_output = np.clip(final_output.astype(np.int16) + noise, 0, 255).astype(np.uint8)

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

            print(f"[{class_id+1}/7] {class_name.upper()} - Generated: {file_identifier}.jpg | Set: {assignment}")

    print(f"Dataset successfully compiled inside parent directory: '{DATASET_DIR}'!")

if __name__ == "__main__":
    build_dataset(volume_per_class=250, split=0.8)