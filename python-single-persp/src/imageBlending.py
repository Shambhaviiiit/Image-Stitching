import numpy as np
import cv2

def image_blending(warped_img1, warped_img2, blend_type):
    # Convert to grayscale and binarize
    w1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
    w2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    _, w1 = cv2.threshold(w1, 0, 255, cv2.THRESH_BINARY)
    _, w2 = cv2.threshold(w2, 0, 255, cv2.THRESH_BINARY)

    # Fill holes in the binarized images
    w1 = cv2.morphologyEx(w1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    w2 = cv2.morphologyEx(w2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Normalize the weights
    w1 = w1 / 255.0
    w2 = w2 / 255.0

    if blend_type == 'average':
        # Average blending
        output_canvas = np.zeros_like(warped_img1)
        for i in range(3):  # For each channel
            output_canvas[:, :, i] = (warped_img1[:, :, i] * w1 + warped_img2[:, :, i] * w2) / (w1 + w2)

    elif blend_type == 'linear':
        # Linear blending
        out = warped_img1
        out_mask = w1
        
        # Find centers of masks
        r1, c1 = np.where(w1 > 0)
        out_center1 = [np.mean(r1), np.mean(c1)]

        r2, c2 = np.where(w2 > 0)
        out_center2 = [np.mean(r2), np.mean(c2)]

        # Compute weighted mask
        vec = np.array(out_center2) - np.array(out_center1)
        intsct_mask = np.logical_and(w1 > 0, w2 > 0)
        r, c = np.where(intsct_mask)

        proj_val = (r - out_center1[0]) * vec[0] + (c - out_center1[1]) * vec[1]
        out_wmask = np.zeros_like(w1)
        proj_val = (proj_val - (np.min(proj_val) + 1e-3)) / (np.max(proj_val) - (np.min(proj_val) + 1e-3))
        out_wmask[r, c] = proj_val

        # Create the blending masks
        mask1 = np.stack([out_mask] * 3, axis=-1) & (out_wmask == 0)
        mask2 = np.stack([out_wmask] * 3, axis=-1)
        mask3 = np.stack([w2] * 3, axis=-1) & (out_wmask == 0)

        # Perform the blending
        output_canvas = out * (mask1 + (1 - mask2) * (mask2 != 0)) + warped_img2 * (mask2 + mask3)

    return output_canvas
