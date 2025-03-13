import numpy as np

class OCRBubble:
    """
    A class to represent a bubble detected in an image.
    It's used to store the information of a bubble detected in an image.
    It's the output of the FullProcessingThread class and the OnlyOCRProcessingThread class.

    Attributes:
        position (tuple): The position of the bubble on the image.
        boundingBox (list): The bounding box of the bubble.
        whiteMask (np.ndarray): The white mask of the bubble.
        ocrText (str): The text extracted from the bubble using OCR.
        bubbleText (str): The text extracted from the bubble using the bubble detection model.
    """

    def __init__(self, position:tuple,boundingBox:list, whiteMask, ocrText:str, bubbleText:str):
        """
        Args:
            position (tuple): The position of the bubble on the image.
            boundingBox (list): The bounding box of the bubble.
            whiteMask (np.ndarray): The white mask of the bubble.
            ocrText (str): The text extracted from the bubble using OCR.
            bubbleText (str): The text extracted from the bubble using the bubble detection model.
        Returns:
            None
        """
        assert (isinstance(position, tuple) or position is None), "Position must be a tuple."
        assert (isinstance(boundingBox, list) or boundingBox is None), "BoundingBox must be a list."
        assert (isinstance(whiteMask, np.ndarray) or whiteMask is None), "WhiteMask must be a NumPy array."
        assert (isinstance(ocrText, str) or ocrText is None), "OCRText must be a string."
        assert (isinstance(bubbleText, str) or bubbleText is None), "BubbleText must be a string."

        self.__position = position
        self.__boundingBox = boundingBox
        self.__whiteMask = whiteMask
        self.__ocrText = ocrText
        self.__bubbleText = bubbleText

    #region Getters
    def getPosition(self):
        return self.__position
    
    def getBoundingBox(self):
        return self.__boundingBox
    
    def getWhiteMask(self):
        return self.__whiteMask
    
    def getOCRText(self):
        return self.__ocrText
    
    def getBubbleText(self):
        return self.__bubbleText
    #endregion

    #region Setters
    def setPosition(self, position:tuple):
        self.__position = position

    def setBoundingBox(self, boundingBox:list):
        self.__boundingBox = boundingBox
    
    def setWhiteMask(self, whiteMask:np.array):
        self.__whiteMask = whiteMask
    
    def setOCRText(self, ocrText:str):
        self.__ocrText = ocrText

    def setBubbleText(self, bubbleText:str):
        self.__bubbleText = bubbleText
    #endregion

    def isPointInside(self, point:tuple):
        """
        Check if a point is inside the bubble's bounding box.
        
        Args:
            point (tuple): The point to check.
        
        Returns:
            bool: True if the point is inside the bounding box, False otherwise.
        """
        if self.__boundingBox is None:
            return False
        
        x, y = point
        x_min, y_min, x_max, y_max = self.__boundingBox
        return x_min <= x <= x_max and y_min <= y <= y_max

    def serialize(self, mask_storage='cropped_rle'):
        """
        Serialize the OCRBubble object with optimized storage.
        
        Args:
            mask_storage (str): Storage method for the mask:
                - 'none': Don't store the mask, just bounding box
                - 'cropped_rle': Run-length encode only the cropped region (default, most efficient)
                - 'ellipse': Store as ellipse parameters (very efficient for oval bubbles)
                - 'contour_delta': Store contour using delta encoding
                - 'rle': Standard run-length encoding
                - 'full': Store the full mask array (largest files)
        
        Returns:
            A dictionary containing the serialized OCRBubble object.
        """
        # Handle the mask based on storage method
        serialized_mask = None
        
        if self.__whiteMask is not None and self.__boundingBox is not None:
            if mask_storage == 'none':
                # Don't store mask, just use bounding box for reconstruction
                serialized_mask = None
                
            elif mask_storage == 'cropped_rle':
                # Most efficient: RLE encoding of just the cropped region
                mask = self.__whiteMask[0] > 0  # Convert to binary
                x_min, y_min, x_max, y_max = self.__boundingBox
                
                # Crop the mask to bounding box
                cropped = mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                # Compress with optimized RLE
                import zlib
                
                # Convert binary mask to bytes (0s and 1s)
                mask_bytes = np.packbits(cropped)
                
                # Compress with zlib
                compressed = zlib.compress(mask_bytes)
                
                serialized_mask = {
                    'type': 'cropped_rle',
                    'shape': list(cropped.shape),
                    'data': list(compressed),  # Convert bytes to list for JSON
                    'origin': [int(x_min), int(y_min)]
                }
                
            elif mask_storage == 'contour_delta':
                # Delta encoding for contours (stores differences between points)
                from skimage import measure
                
                mask = self.__whiteMask[0] > 0
                contours = measure.find_contours(mask, 0.5)
                
                if len(contours) > 0:
                    contour = max(contours, key=len)
                    # Quantize to integers and reduce points
                    points = contour.astype(int)[::3]  # Take every 3rd point
                    
                    # Delta encoding - store differences between consecutive points
                    y_deltas = []
                    x_deltas = []
                    
                    if len(points) > 0:
                        # Store first point as is
                        y0, x0 = points[0]
                        y_deltas.append(int(y0))  # First y coordinate
                        x_deltas.append(int(x0))  # First x coordinate
                        
                        # Store subsequent points as deltas
                        prev_y, prev_x = y0, x0
                        for y, x in points[1:]:
                            y_deltas.append(int(y - prev_y))
                            x_deltas.append(int(x - prev_x))
                            prev_y, prev_x = y, x
                    
                    serialized_mask = {
                        'type': 'contour_delta',
                        'shape': list(mask.shape),
                        'y_deltas': y_deltas,
                        'x_deltas': x_deltas
                    }
                
            elif mask_storage == 'rle':
                # Standard RLE (already implemented, keep as fallback)
                # [existing implementation]
                mask = self.__whiteMask[0] > 0
                runs = []
                count = 0
                last = False
                
                for pixel in mask.flatten():
                    if pixel == last:
                        count += 1
                    else:
                        runs.append(count)
                        count = 1
                        last = pixel
                runs.append(count)
                
                serialized_mask = {
                    'type': 'rle',
                    'shape': list(mask.shape),
                    'runs': runs,
                    'start_value': not bool(mask.flatten()[0])
                }
                
            elif mask_storage == 'full':
                # Full mask (included for completeness)
                serialized_mask = {
                    'type': 'full',
                    'data': self.__whiteMask.tolist()
                }
        
        # Ensure all values are JSON serializable
        box = [int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x 
            for x in self.__boundingBox] if self.__boundingBox else None
        
        return {
            "position": list(self.__position) if self.__position else None,
            "boundingBox": box,
            "whiteMask": serialized_mask,
            "ocrText": self.__ocrText,
            "bubbleText": self.__bubbleText
        }
    
    def __str__(self):
        """
        Return a string representation of the OCRBubble object.
        
        Returns:
            str: A string representation of the OCRBubble object.
        """
        return f"OCRBubble(position={self.__position}, boundingBox={self.__boundingBox}, ocrText='{self.__ocrText}', bubbleText='{self.__bubbleText}')"

    @staticmethod
    def fromJSON(data):
        """
        Create an OCRBubble object from serialized JSON data.
        
        Args:
            data (dict): Dictionary containing serialized OCRBubble data from JSON file
            
        Returns:
            OCRBubble: A new OCRBubble instance
        """
        # Extract basic attributes
        position = tuple(data.get("position")) if data.get("position") else None
        bounding_box = data.get("boundingBox")
        ocr_text = data.get("ocrText")
        bubble_text = data.get("bubbleText")
        
        # Reconstruct the mask based on storage type
        mask = None
        mask_data = data.get("whiteMask")
        
        if mask_data and bounding_box:
            mask_type = mask_data.get("type")
            
            if mask_type == "cropped_rle":
                # Decompress cropped RLE mask
                import zlib
                
                shape = mask_data.get("shape")
                origin = mask_data.get("origin")
                
                # Convert the list back to bytes
                compressed_bytes = bytes(mask_data.get("data"))
                
                # Decompress with zlib
                decompressed = zlib.decompress(compressed_bytes)
                
                # Unpack bits to binary array
                unpacked = np.unpackbits(np.frombuffer(decompressed, dtype=np.uint8))
                # Trim or pad to match expected size (needed because packbits pads to multiple of 8)
                expected_size = shape[0] * shape[1]
                unpacked = unpacked[:expected_size]
                
                # Reshape to original dimensions
                cropped_mask = unpacked.reshape(shape).astype(bool)
                
                # Create full-size mask and place cropped region at original position
                full_height = max(origin[1] + shape[0], bounding_box[3]) 
                full_width = max(origin[0] + shape[1], bounding_box[2])
                
                full_mask = np.zeros((3, full_height, full_width), dtype=np.uint8)
                full_mask[0, origin[1]:origin[1]+shape[0], origin[0]:origin[0]+shape[1]] = cropped_mask * 255
                
                mask = full_mask
            
            elif mask_type == "contour_delta":
                # Reconstruct from delta encoding
                from skimage.draw import polygon
                
                shape = mask_data.get("shape")
                y_deltas = mask_data.get("y_deltas")
                x_deltas = mask_data.get("x_deltas")
                
                # Reconstruct points from deltas
                y_points = []
                x_points = []
                
                # First point as is
                prev_y = y_deltas[0]
                prev_x = x_deltas[0]
                y_points.append(prev_y)
                x_points.append(prev_x)
                
                # Reconstruct subsequent points
                for i in range(1, len(y_deltas)):
                    prev_y += y_deltas[i]
                    prev_x += x_deltas[i]
                    y_points.append(prev_y)
                    x_points.append(prev_x)
                
                # Create mask
                mask_2d = np.zeros(shape, dtype=np.uint8)
                rr, cc = polygon(y_points, x_points, shape)
                mask_2d[rr, cc] = 255
                
                # Create 3D mask
                mask = np.zeros((3, shape[0], shape[1]), dtype=np.uint8)
                mask[0] = mask_2d
                
            elif mask_type == "rle":
                # Reconstruct from run-length encoding
                
                shape = mask_data.get("shape")
                runs = mask_data.get("runs")
                start_value = mask_data.get("start_value")
                
                # Allocate flat array
                flat_size = shape[0] * shape[1]
                mask_flat = np.zeros(flat_size, dtype=np.uint8)
                
                # Fill with RLE data
                idx = 0
                value = 255 if start_value else 0  # Start with 255 if start_value is True
                
                for run_length in runs:
                    end_idx = min(idx + run_length, flat_size)
                    mask_flat[idx:end_idx] = value
                    value = 0 if value == 255 else 255  # Toggle between 0 and 255
                    idx = end_idx
                    if idx >= flat_size:
                        break
                
                # Reshape to 2D then create 3D mask
                mask_2d = mask_flat.reshape(shape)
                mask = np.zeros((3, shape[0], shape[1]), dtype=np.uint8)
                mask[0] = mask_2d
                
            elif mask_type == "full":
                # Direct conversion from full data
                mask = np.array(mask_data.get("data"))
            
            # If no valid mask was created but we have a bounding box, create a rectangular mask
            if mask is None and bounding_box:
                x_min, y_min, x_max, y_max = bounding_box
                height = int(y_max - y_min + 1)
                width = int(x_max - x_min + 1)
                
                # Create empty mask with appropriate size
                img_height = max(int(y_max) + 1, height + int(y_min))
                img_width = max(int(x_max) + 1, width + int(x_min))
                
                mask = np.zeros((3, img_height, img_width), dtype=np.uint8)
                mask[0, int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 255
        
        # Create and return the OCRBubble object
        return OCRBubble(position, bounding_box, mask, ocr_text, bubble_text)


