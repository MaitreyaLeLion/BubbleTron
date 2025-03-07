from PyQt6.QtWidgets import QGraphicsPixmapItem

class OCRBubble:
    """
    A class to represent a bubble detected in an image.
    It's used to store the information of a bubble detected in an image.
    It's the output of the FullProcessingThread class and the OnlyOCRProcessingThread class.

    Attributes:
        position (tuple): The position of the bubble on the image.
        boundingBox (list): The bounding box of the bubble.
        whiteMask (QGraphicsPixmapItem): The white mask of the bubble.
        ocrText (str): The text extracted from the bubble using OCR.
        bubbleText (str): The text extracted from the bubble using the bubble detection model.
    """

    def __init__(self, position:tuple,boundingBox:list, whiteMask, ocrText:str, bubbleText:str):
        """
        Args:
            position (tuple): The position of the bubble on the image.
            boundingBox (list): The bounding box of the bubble.
            whiteMask (QGraphicsPixmapItem): The white mask of the bubble.
            ocrText (str): The text extracted from the bubble using OCR.
            bubbleText (str): The text extracted from the bubble using the bubble detection model.
        Returns:
            None
        """
        assert (isinstance(position, tuple) or position is None), "Position must be a tuple."
        assert (isinstance(boundingBox, list) or boundingBox is None), "BoundingBox must be a list."
        assert (isinstance(whiteMask, QGraphicsPixmapItem) or whiteMask is None), "WhiteMask must be a QGraphicsPixmapItem."
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
    
    def setWhiteMask(self, whiteMask:QGraphicsPixmapItem):
        self.__whiteMask = whiteMask
    
    def setOCRText(self, ocrText:str):
        self.__ocrText = ocrText

    def setbubbleText(self, bubbleText:str):
        self.__bubbleText = bubbleText
    #endregion


