############################################
# FluoroAnalyzer
# Author: Peiyao (Pearl) Liu <Peiyao.Liu@cchmc.org>
# Date: 20250120
# Aim: Analyze transfection efficiency based on brightfield and GFP images.
############################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load images
brightfield_path = '/Users/Pearl/Projects/Bap1_ZYLab_2403_present/Bap1_ESCs_KO/ESCs_BF_4X.JPG'
gfp_path = '/Users/Pearl/Projects/Bap1_ZYLab_2403_present/Bap1_ESCs_KO/ESCs_GFP_4X.JPG'
brightfield_image = Image.open(brightfield_path).convert('L')
gfp_image = Image.open(gfp_path).convert('L')

# Convert to numpy arrays
brightfield_array = np.array(brightfield_image)
gfp_array = np.array(gfp_image)

# Thresholds
brightfield_threshold = 180  # Manually set threshold for brightfield
gfp_threshold = 40           # Manually set threshold for GFP

# Binary masks
brightfield_cells = brightfield_array > brightfield_threshold  # Detect cells in brightfield
gfp_positive_cells = brightfield_cells & (gfp_array > gfp_threshold)  # Detect GFP-positive cells within brightfield cells

# Calculate results
total_cells = np.sum(brightfield_cells)  # Total cells detected in brightfield
gfp_positive_count = np.sum(gfp_positive_cells)  # GFP-positive cells
transfection_efficiency = (gfp_positive_count / total_cells) * 100  # Transfection efficiency

# Combined visualization
combined_image = np.zeros((*brightfield_array.shape, 3), dtype=np.uint8)
combined_image[..., 0] = brightfield_array  # Red channel for brightfield
combined_image[..., 1] = (gfp_positive_cells * 255).astype(np.uint8)  # Green channel for GFP-positive

# Plot results
plt.figure(figsize=(18, 12))

# Original Brightfield
plt.subplot(2, 3, 1)
plt.title("Brightfield Image (Original)")
plt.imshow(brightfield_array, cmap='gray')
plt.axis('off')

# Original GFP
plt.subplot(2, 3, 2)
plt.title("GFP Image (Original)")
plt.imshow(gfp_array, cmap='gray')
plt.axis('off')

# Intensity Distribution
plt.subplot(2, 3, 3)
plt.hist(brightfield_array.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7, label='Brightfield')
plt.hist(gfp_array.ravel(), bins=256, range=(0, 255), color='green', alpha=0.7, label='GFP')
plt.axvline(brightfield_threshold, color='blue', linestyle='--', label=f'Brightfield Threshold: {brightfield_threshold}')
plt.axvline(gfp_threshold, color='orange', linestyle='--', label=f'GFP Threshold: {gfp_threshold}')
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()

# Thresholded Brightfield Cells
plt.subplot(2, 3, 4)
plt.title("Thresholded Brightfield Cells")
plt.imshow(brightfield_cells, cmap='gray')
plt.axis('off')
plt.text(10, 10, f"Total Cells: {total_cells}", color='red', fontsize=12, ha='left', va='top')

# Thresholded GFP Positive Cells
plt.subplot(2, 3, 5)
plt.title("Thresholded GFP Positive Cells")
plt.imshow(gfp_positive_cells, cmap='gray')
plt.axis('off')
plt.text(10, 10, f"GFP+ Cells: {gfp_positive_count}", color='red', fontsize=12, ha='left', va='top')

# Combined Visualization
plt.subplot(2, 3, 6)
plt.title("Combined Brightfield + GFP")
plt.imshow(combined_image)
plt.axis('off')
plt.text(10, 10, f"Transfection Efficiency: {transfection_efficiency:.2f}%", color='white', fontsize=12, ha='left', va='top')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2, wspace=0.3, hspace=0.05)
plt.show()

# Print results
print(f"Total Cells Detected in Brightfield: {total_cells}")
print(f"GFP-Positive Cells Detected: {gfp_positive_count}")
print(f"Transfection Efficiency: {transfection_efficiency:.2f}%")

# Save the figure to a file
plt.savefig('output_transfection_analysis.png', dpi=300, bbox_inches='tight')
