from time import perf_counter
import numpy as np
import asyncio
from PIL import Image
from PyQt6.QtCore import QThread
import deepl
from utils.utils import verbosePrint
from utils.OCRBubble import OCRBubble

class OnlyOCRProcessingThread(QThread):

    numberOfThreadWithOffset = 500000
    def __init__(self, mainWindow, imagePath, mask, verboseLevel=0):
        super().__init__()
        self.verboseLevel = verboseLevel
        self.mainWindow = mainWindow
        self.imagePath = imagePath
        self.mask = mask
        self.threadId = OnlyOCRProcessingThread.numberOfThreadWithOffset
        OnlyOCRProcessingThread.numberOfThreadWithOffset += 1
        self.is_running = True
        self.api_key = "17a4ea22-137d-441e-8399-3d18266173a6:fx"

        self.startingTime = perf_counter()
        self.output = OCRBubble(position=None, boundingBox=None, whiteMask=None, ocrText=None, bubbleText=None)
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def translate_text(self, text):
        """Translate text from Japanese to English using DeepL API.
        Args:
            text: A string of text to translate.
        Returns:
            A string of translated text.
        """
        deepl_client = deepl.DeepLClient(self.api_key, send_platform_info=False)
        translation = deepl_client.translate_text(text, target_lang="EN-US", source_lang="JA",glossary="5ee17871-824b-4d69-a6a9-156a9fb39e50")
        if self.mainWindow.targetLang != "EN-US":
            translation = deepl_client.translate_text(translation.text, target_lang=self.mainWindow.targetLang, source_lang="EN")
        
        return translation.text

    def run(self):
        verbosePrint(text="Running OCR.", verboseLevel=self.verboseLevel, verboseThreshold=2)
        t0 = perf_counter()
        image = Image.open(self.imagePath).convert('RGB')
        image_np = np.array(image)
        mask2d = self.mask[0]

        # Get mask bounds
        rows = np.any(mask2d, axis=1)
        cols = np.any(mask2d, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        self.output.setBoundingBox([cmin, rmin, cmax, rmax])
        self.output.setPosition((cmin, rmin))
        
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
            verbosePrint(f"Translation error: {e}")
            # self.mainWindow.OCRTextQueue.append((self.threadId, text))
            self.output.setbubbleText(text)
        finally:
            self.loop.close()
        
        self.mainWindow.totalProcessTime += perf_counter()-self.startingTime
        self.mainWindow.predictionOutputQueue.put(self.output)

        verbosePrint(text=f"Total time taken for prediction thread {self.threadId} : {perf_counter()-self.startingTime}s.", verboseLevel=self.verboseLevel, verboseThreshold=1)
    
    def stop(self):
        self.is_running = False
        if self.loop and self.loop.is_running():
            self.loop.stop()

