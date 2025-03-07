import numpy as np

def verbosePrint(text, verboseLevel, verboseThreshold):
    """Print text if verbose level is above the threshold.
    Args:
        text: A string of text to print.
        verboseLevel: An integer representing the verbose level.
        verboseThreshold: An integer representing the verbose threshold.
    """
    if verboseLevel > verboseThreshold:
        print(text)

def boundingBoxFromMask(mask):
        """Find the bounding box conscripted in the mask.
        Args:
            mask: A 2D or 3D NumPy array where the first dimension is the channel.
        Returns:
            A tuple of integers (left, top, right, bottom) representing the bounding box.
        """
        # Ensure the mask is a 2D array
        if mask.ndim == 3:
            gray_mask = mask[0, :, :]
        elif mask.ndim == 2:
            gray_mask = mask
        else:
            raise ValueError("Mask must be a 2D or 3D NumPy array.")

        # Get mask dimensions
        height, width = gray_mask.shape
        
        # Find all non-zero points
        y_coords, x_coords = np.nonzero(gray_mask)
        if len(x_coords) == 0:
            return None
            
        
        # Precompute vertical fills for the whole mask
        # This avoids recomputing for each x-range combination
        vertical_fills = np.zeros((width+1, height), dtype=bool)
        for i in range(width):
            if i == 0:
                vertical_fills[i] = gray_mask[:, i] > 0
            else:
                vertical_fills[i] = vertical_fills[i-1] & (gray_mask[:, i] > 0)
        
        # Initialize variables for the best box
        best_area = 0
        best_box = None
        width_factor = 1.5  # Prioritize width over height
        
        # Import groupby/itemgetter only once
        from itertools import groupby
        from operator import itemgetter
        
        # Reduced sampling for large masks
        max_unique_points = 200
        unique_x = np.unique(x_coords)
        if len(unique_x) > max_unique_points:
            step = len(unique_x) // max_unique_points
            unique_x = unique_x[::step]
        
        # For each possible left edge
        for left in unique_x:
            # For each possible right edge to the right of left edge
            rights = unique_x[unique_x > left]
            if len(rights) > max_unique_points:
                step = len(rights) // max_unique_points
                rights = rights[::step]
                
            for right in rights:
                # Look up the precomputed vertical fills
                if left == 0:
                    fills = vertical_fills[right]
                else:
                    # For each column between left and right, all pixels must be filled
                    fills = np.all(gray_mask[:, left:right+1] > 0, axis=1)
                
                if not np.any(fills):
                    continue
                    
                # Find continuous ranges of True values
                ranges = np.where(fills)[0]
                
                # Group consecutive numbers into ranges more efficiently
                ranges_grouped = []
                for k, g in groupby(enumerate(ranges), lambda x: x[0] - x[1]):
                    group = list(map(itemgetter(1), g))
                    ranges_grouped.append((group[0], group[-1]))
                
                # Calculate area for each range and keep track of the best
                for y_min_range, y_max_range in ranges_grouped:
                    area = (right - left + 1) * (y_max_range - y_min_range + 1) * width_factor
                    if area > best_area:
                        best_area = area
                        best_box = (left, y_min_range, right, y_max_range)
        
        return best_box

def computeOptimalFontSize(font, text, box_width, box_height, painter):
    # Auto-size calculation parameters
    min_font_size = 8
    max_font_size = 72
    padding = 4  # Pixels of padding around text
    
    # Binary search to find optimal font size
    low = min_font_size
    high = max_font_size
    optimal_size = min_font_size
    optimal_lines = []
    
    while low <= high:
        mid = (low + high) // 2
        font.setPointSize(mid)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        
        # Calculate usable width accounting for padding
        usable_width = box_width - (padding * 2)
        
        # Split text to check how it fits with this font size
        words = text.split()
        lines = []
        current_line = ""
        
        # Text wrapping simulation
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if metrics.horizontalAdvance(test_line) <= usable_width:
                current_line = test_line
            else:
                # If a single word is too wide for the box, this font size is too large
                if not current_line and metrics.horizontalAdvance(word) > usable_width:
                    high = mid - 1
                    break
                
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        # Add the last line if any
        if current_line:
            lines.append(current_line)
        
        # Calculate total text height with padding
        total_height = len(lines) * metrics.height() + (padding * 2)
        
        # Find maximum line width to verify horizontal fit
        max_line_width = 0
        if lines:
            max_line_width = max(metrics.horizontalAdvance(line) for line in lines)
        
        # Check if text fits in the box with proper margins
        fits_horizontally = max_line_width <= usable_width
        fits_vertically = total_height <= box_height
        
        if fits_horizontally and fits_vertically and len(lines) > 0:
            optimal_size = mid
            optimal_lines = lines.copy()  # Save these lines for later use
            low = mid + 1  # Try a larger size
        else:
            high = mid - 1  # Try a smaller size

    return optimal_size, optimal_lines, padding

