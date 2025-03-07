from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem
import numpy as np
import asyncio
import deepl
from time import perf_counter
from PIL import Image
from utils.utils import verbosePrint, boundingBoxFromMask
from utils.OCRBubble import OCRBubble

class FullProcessingThread(QThread):
    numberOfThreads = 0

    def __init__(self, mainWindow, imagePath, points=None, labels=None, box=None, verboseLevel=0):
        super().__init__()
        self.verboseLevel = verboseLevel
        self.mainWindow = mainWindow
        self.imagePath = imagePath
        self.points = points
        self.labels = labels
        self.threadId = FullProcessingThread.numberOfThreads
        FullProcessingThread.numberOfThreads += 1
        self.api_key = "17a4ea22-137d-441e-8399-3d18266173a6:fx"
        self.startingTime = perf_counter()
        self.output = OCRBubble(position=None, boundingBox=None, whiteMask=None, ocrText=None, bubbleText=None)

        # SAM2 prediction box
        self.box = np.array(box) if box else None
        if self.box is not None:
            self.box = self.box[None, :]

        # Async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.is_running = True

    async def translate_text(self, text):
        """Translate text from Japanese to English using DeepL API.
        Args:
            text: A string of text to translate.
        Returns:
            A string of translated text.
        """
        # deepl_client = deepl.DeepLClient(self.api_key, send_platform_info=False)
        # glossary = "5ee17871-824b-4d69-a6a9-156a9fb39e50" if  self.mainWindow.targetLang == "EN-US" else None
        # translation = deepl_client.translate_text(text, target_lang=self.mainWindow.targetLang, source_lang="JA",glossary=glossary)

        deepl_client = deepl.DeepLClient(self.api_key, send_platform_info=False)
        translation = deepl_client.translate_text(text, target_lang="EN-US", source_lang="JA",glossary="5ee17871-824b-4d69-a6a9-156a9fb39e50")
        if self.mainWindow.targetLang != "EN-US":
            translation = deepl_client.translate_text(translation.text, target_lang=self.mainWindow.targetLang, source_lang="EN")
        
        return translation.text

    def run(self):
        verbosePrint("Running SAM2 prediction.", self.verboseLevel, 1)
        t0 = perf_counter()
        #self.imagePath, self.points, self.labels, self.box
        verbosePrint(text=f"Image path : {self.imagePath};\nPoints : {self.points};\nLabels : {self.labels}\nBox : {self.box};", verboseLevel=self.verboseLevel, verboseThreshold=2)
        output_mask = self.mainWindow.bd.predict(self.imagePath, self.points, self.labels, self.box)
        verbosePrint(text=f"Time taken for SAM2 prediction : {perf_counter()-t0}s.", verboseLevel=self.verboseLevel, verboseThreshold=1)

        mask = output_mask[0] # outputMask is a tuple of mask, score, we only need the mask
        
        # Include thread ID when putting mask in queue for tracking
        # self.mainWindow.numpyArraySAMMaskOutputQueue.put((self.threadId, mask))
        # self.output["numpyArraySAMMask"] = mask
        
        _, height, width = mask.shape # Get mask dimensions
        
        verbosePrint(text="Running calculations to extract bounding box from mask.", verboseLevel=self.verboseLevel, verboseThreshold=2)
        t0 = perf_counter()
        box = boundingBoxFromMask(mask)
        # Store box with thread ID for tracking
        # self.mainWindow.textBoxesQueue.append((self.threadId, box))
        self.output.setBoundingBox(box)
        self.output.setPosition((box[0], box[1]))
        verbosePrint(text=f"Time taken for bounding box extraction : {perf_counter()-t0}s.", verboseLevel=self.verboseLevel, verboseThreshold=2)

        verbosePrint(text="Converting mask to QImage.", verboseLevel=self.verboseLevel, verboseThreshold=2)
        t0 = perf_counter()

        # Create RGBA image with white color and alpha channel
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[..., :3] = 255  # Set RGB channels to white (255)
        rgba_image[..., 3] = np.where(mask[0] > 0, 255, 0)  # Set alpha: 255 for mask, 0 for background

        # Only create border if drawBorder is True
        if False:
            verbosePrint(text="Adding border to mask.", verboseLevel=self.verboseLevel, verboseThreshold=2)
            
            # Create a border by finding edge pixels
            mask_2d = mask[0].copy()
                
            # Create a slightly eroded version of the mask to find the border
            from scipy import ndimage
            eroded_mask = ndimage.binary_erosion(mask_2d, iterations=2)
            border_mask = np.logical_and(mask_2d, ~eroded_mask)  # Border pixels

            # Set border pixels to black (0,0,0) while keeping alpha at 255
            rgba_image[..., 0][border_mask] = 0  # R
            rgba_image[..., 1][border_mask] = 0  # G
            rgba_image[..., 2][border_mask] = 0  # B
            # Alpha is already set to 255 for these pixels

        # Create QImage from RGBA data
        qimage = QImage(rgba_image.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        verbosePrint(text=f"Time taken for converting mask to QImage : {perf_counter()-t0}s.", verboseLevel=self.verboseLevel, verboseThreshold=2)
        mask_pixmap = QPixmap.fromImage(qimage)
        mask_item = QGraphicsPixmapItem(mask_pixmap)
        # Include thread ID when putting item in queue
        # self.mainWindow.SAMMaskOutputQueue.put((self.threadId, mask_item))
        self.output.setWhiteMask(mask_item) 
        
        verbosePrint(text="Running OCR.", verboseLevel=self.verboseLevel, verboseThreshold=2)
        t0 = perf_counter()
        image = Image.open(self.imagePath).convert('RGB')
        image_np = np.array(image)
        mask2d = mask[0]

        # Get mask bounds
        rows = np.any(mask2d, axis=1)
        cols = np.any(mask2d, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop and process image
        cropped_image = image_np[rmin:rmax+1, cmin:cmax+1]
        cropped_mask = mask2d[rmin:rmax+1, cmin:cmax+1]
        masked_image = np.where(cropped_mask[..., None] > 0, cropped_image, [0,0,0])
        masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))
        
        # Perform OCR
        text = self.mainWindow.mocr(masked_image_pil)
        self.output.setOCRText(text)
        verbosePrint(text=f"OCR text : {text}", verboseLevel=self.verboseLevel, verboseThreshold=1)
        # text = text.replace("♡", " ")
        # text = text.replace("♪", " ")
        verbosePrint(text=f"Time taken for OCR : {perf_counter()-t0}s.", verboseLevel=self.verboseLevel, verboseThreshold=2)

        verbosePrint(text="Running translation.", verboseLevel=self.verboseLevel, verboseThreshold=2)
        t0 = perf_counter()

        try:
            # Run translation in the event loop
            english_translation = self.loop.run_until_complete(
                self.translate_text(text)
            )
            verbosePrint(text=f"English translation : {english_translation}", verboseLevel=self.verboseLevel, verboseThreshold=1)
            verbosePrint(text=f"Time taken for translation : {perf_counter()-t0}s.", verboseLevel=self.verboseLevel, verboseThreshold=2)
            # Store text with thread ID
            # self.mainWindow.OCRTextQueue.append((self.threadId, english_translation))
            self.output.setbubbleText(english_translation)
        except Exception as e:
            print(f"Translation error: {e}")
            # self.mainWindow.OCRTextQueue.append((self.threadId, text))
            self.output.setbubbleText(text)
        finally:
            self.loop.close()
        
        self.mainWindow.totalProcessTime += perf_counter()-self.startingTime

        self.mainWindow.predictionOutputQueue.put(self.output)

        verbosePrint(text=f"Total time taken for prediction thread {self.threadId} : {perf_counter()-self.startingTime}s.", verboseLevel=self.verboseLevel, verboseThreshold=1)

    def stop(self):
        self.is_running = False
        if self.loop.is_running():
            self.loop.stop()

    