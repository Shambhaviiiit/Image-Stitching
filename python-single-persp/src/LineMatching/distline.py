import numpy as np

def distline(line1, line2):
    """
    Calculate a similarity measure between two line segments.
    
    line1: A tuple or array of (x1, y1, x2, y2) representing the first line segment.
    line2: A tuple or array of (x1, y1, x2, y2) representing the second line segment.
    
    Returns:
    A tuple (simL, simR) representing similarity measures between the two lines.
    """
    # Extract points from the lines
    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2

    # Calculate line lengths
    length1 = np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
    length2 = np.sqrt((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)

    # Calculate midpoint distances (as an example of a similarity metric)
    midpoint1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    midpoint2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    distance_between_midpoints = np.sqrt((midpoint2[0] - midpoint1[0])**2 + (midpoint2[1] - midpoint1[1])**2)

    # Angle similarity (optional)
    angle1 = np.arctan2(y2_1 - y1_1, x2_1 - x1_1)
    angle2 = np.arctan2(y2_2 - y1_2, x2_2 - x1_2)
    angle_diff = np.abs(angle1 - angle2)

    # Normalize and return similarity scores
    simL = 1 - (distance_between_midpoints / max(length1, length2))  # Basic similarity based on distance
    simR = 1 - (angle_diff / np.pi)  # Angle similarity

    # Ensure scores are within [0, 1]
    simL = max(0, min(1, simL))
    simR = max(0, min(1, simR))

    return simL, simR

# Example usage
# line1 = (10, 10, 50, 50)  # (x1, y1, x2, y2)
# line2 = (15, 15, 55, 55)  # (x1, y1, x2, y2)
# similarity = distline(line1, line2)
# print(f"simL: {similarity[0]:.3f}, simR: {similarity[1]:.3f}")
