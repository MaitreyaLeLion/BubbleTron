from time import perf_counter
t0 = perf_counter()
from PyQt6.QtCore import (
    QEvent,
    QRectF,
    Qt,
    QEventLoop,
    pyqtSignal as Signal,
    pyqtSlot as Slot,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QFont,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QFontComboBox,
    QComboBox,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QCheckBox,
    QTextEdit, 
    QDialog, 
    QVBoxLayout, 
    QPushButton
)
import os
import queue
import json
import numpy as np

from utils.utils import computeOptimalFontSize, verbosePrint, createWhiteMaskPixmapFromNpMask, bubbleIsNotInDict
from utils.FullProcessingThread import FullProcessingThread
from utils.OnlyOCRProcessingThread import OnlyOCRProcessingThread
from utils.NumpyEncoder import NumpyEncoder
from utils.OCRBubble import OCRBubble

import utils.BubbleDetector as BD
from ultralytics import YOLO
from manga_ocr import MangaOcr
print("Time taken for imports: ", perf_counter() - t0)


class MainWindow(QMainWindow):  # Changed from QWidget to QMainWindow
    def __init__(self):
        super().__init__()
        self.verbose = False
        t0 = perf_counter()
        self.setWindowTitle("BubbleTron")
        self.setGeometry(200, 200, 800, 600)

        self.verboseLevel = 2

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create a single main horizontal layout
        self.layout = QHBoxLayout(central_widget)

        # Graphics scene
        self.scene = QGraphicsScene()

        # Add main menu
        self.addMainMenu()

        self.targetLang = "EN-US"
        self.targetLangDict = {'Bulgarian': 'BG', 'Czech': 'CS', 'Danish': 'DA', 'German': 'DE', 'Greek': 'EL', 'English (British)': 'EN-GB', 'English (American)': 'EN-US', 'Spanish': 'ES', 'Estonian': 'ET', 'Finnish': 'FI', 'French': 'FR', 'Hungarian': 'HU', 'Indonesian': 'ID', 'Italian': 'IT', 'Japanese': 'JA', 'Korean': 'KO', 'Lithuanian': 'LT', 'Latvian': 'LV', 'Norwegian': 'NB', 'Dutch': 'NL', 'Polish': 'PL', 'Portuguese (Brazilian)': 'PT-BR', 'Portuguese (European)': 'PT-PT', 'Romanian': 'RO', 'Russian': 'RU', 'Slovak': 'SK', 'Slovenian': 'SL', 'Swedish': 'SV', 'Turkish': 'TR', 'Ukrainian': 'UK', 'Chinese (simplified)': 'ZH-HANS'}
        
        # Set up the complete UI with all panels in one horizontal layout
        self.setupUI()

        

        # Scrollbar policies
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Selection variables
        self.isSelecting = False
        self.selectionStart = None
        self.selectionRect = None
        self.shiftPressed = False
        self.ctrlPressed = False

        self.detectionToolSelected = True
        self.textToolSelected = False

        
        # Mouse tracking and event filter
        self.view.viewport().setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        # Data structures
        self.pixmapItem = None
        self.predictionOutputQueue = queue.Queue()
        self.totalProcessTime = 0
        self.displayedMaskItems = []
        self.allDisplayedItems = []
        self.undisplayedItems = []
        self.activeThread = []

        

        # Bubble detection and OCR
        self.bd = BD.BubbleDetector()
        self.mocr = MangaOcr()
        self.yolo = YOLO("./models/YOLOv8/best.pt")

        verbosePrint(text=f"Time taken for initialization: {perf_counter()-t0}s.", verboseLevel=self.verboseLevel, verboseThreshold=1)

    def setupUI(self):
        """Sets up the main UI with panels and graphics view."""
        
        # Create left side panel and its toggle button
        self.leftSidePanel, self.leftToggleButton = self.createLeftSidePanel()
        
        # Create central view area
        viewContainer = QWidget()
        viewLayout = QVBoxLayout(viewContainer)
        viewLayout.setContentsMargins(0, 0, 0, 0)
        self.view = QGraphicsView(self.scene)
        viewLayout.addWidget(self.view)
        
        # Create right side panel and its toggle button
        self.rightSidePanel, self.rightToggleButton = self.createRightSidePanel()
        
        # Add all components to the main layout
        self.layout.addWidget(self.leftSidePanel)
        self.layout.addWidget(self.leftToggleButton)
        self.layout.addWidget(viewContainer, 1)  # Give the view a stretch factor of 1
        self.layout.addWidget(self.rightToggleButton)
        self.layout.addWidget(self.rightSidePanel)
        
        # Initially hide right panel
        self.rightSidePanel.hide()
        self.rightSidePanelVisible = False
        
        # Left panel is initially visible
        self.leftSidePanelVisible = True

        # Apply the cyberpunk OperaGX-inspired theme
        self.applyCyberpunkTheme()

    def applyCyberpunkTheme(self):
        """Apply a cyberpunk theme inspired by OperaGX to the application."""
        
        # Define the main style sheet
        with open("style/style.txt", "r") as f:
            style_sheet = f.read()
        
        # Apply the style sheet to the application
        self.setStyleSheet(style_sheet)
        
        # Set object names for specific styling
        self.leftToggleButton.setObjectName("toggleButton")
        self.rightToggleButton.setObjectName("toggleButton")
        self.leftSidePanel.setObjectName("leftSidePanel")
        self.rightSidePanel.setObjectName("rightSidePanel")
        self.detectionButton.setObjectName("toolButton")
        self.textEditButton.setObjectName("toolButton")
        
        # Create glow effect for panels
        glow = QGraphicsDropShadowEffect(self)
        glow.setColor(QColor("#FF00FF"))
        glow.setOffset(0, 0)
        glow.setBlurRadius(20)
        self.view.setGraphicsEffect(glow)
        
        # Set viewport properties separately (needed for QGraphicsView)
        self.view.viewport().setStyleSheet("background-color: #121212; border: none;")
        self.scene.setBackgroundBrush(QColor("#121212"))

    def addMainMenu(self):
        """Adds the main menu bar to the window."""

        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # File Menu
        file_menu = QMenu("&File", self)

        # File Actions
        open_action = QAction("&Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.openFunction)
        file_menu.addAction(open_action)

        open_dir_action = QAction("&Open Directory", self)
        open_dir_action.triggered.connect(self.openDirectory)
        file_menu.addAction(open_dir_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.saveFunction)
        file_menu.addAction(save_action)

        exit_action = QAction("&Close", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Close)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        menu_bar.addMenu(file_menu)

        # Edit Menu
        edit_menu = QMenu("&Edit", self)
        copy_action = QAction("&Copy", self)
        edit_menu.addAction(copy_action)
        menu_bar.addMenu(edit_menu)

        # View Menu
        view_menu = QMenu("&View", self)
        menu_bar.addMenu(view_menu)

        # Help Menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)

    def createLeftSidePanel(self):
        """Creates the left side panel with tool selection buttons."""
        # Create side panel
        leftPanel = QWidget()
        leftPanel.setFixedWidth(200)
        sideLayout = QVBoxLayout(leftPanel)
        sideLayout.setContentsMargins(10, 10, 10, 10)  # Add proper margin all around
        
        # Add a grid of buttons for switching between tools
        toolButtonGrid = QWidget()
        toolButtonGridLayout = QVBoxLayout(toolButtonGrid)
        toolButtonGridLayout.setContentsMargins(0, 0, 0, 0)  # Remove internal margins
        toolButtonGridLayout.setSpacing(8)  # Add spacing between buttons
        
        # Detection tool button
        self.detectionButton = QPushButton("Detect")
        self.detectionButton.setCheckable(True)  # Make it checkable
        self.detectionButton.setChecked(True)    # Start with this selected
        self.detectionButton.clicked.connect(self.toggleDetectionTool)
        toolButtonGridLayout.addWidget(self.detectionButton)
        
        # Text editing tool button
        self.textEditButton = QPushButton("Text")
        self.textEditButton.setCheckable(True)   # Make it checkable
        self.textEditButton.clicked.connect(self.toggleTextTool)
        toolButtonGridLayout.addWidget(self.textEditButton)
        
        # Add the tool button grid to the side panel
        sideLayout.addWidget(toolButtonGrid)
        
        # Add stretch to push widgets to top
        sideLayout.addStretch()
        
        # Create toggle button (separate from the panel)
        toggleButton = QPushButton("◀")  # Left arrow
        toggleButton.setFixedWidth(20)
        toggleButton.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        toggleButton.clicked.connect(self.toggleLeftSidePanel)
        
        return leftPanel, toggleButton

    def createRightSidePanel(self):
        """Creates the right side panel with settings controls."""
        # Create side panel
        rightPanel = QWidget()
        rightPanel.setFixedWidth(200)
        sideLayout = QVBoxLayout(rightPanel)
        
        # Add target language selector
        targetLangLabel = QLabel("Target Language:")
        self.targetLangSelector = QComboBox()
        for key in self.targetLangDict.keys():
            self.targetLangSelector.addItem(key)
        self.targetLangSelector.setCurrentText("English (American)")
        self.targetLangSelector.currentTextChanged.connect(self.updateTargetLang)
        sideLayout.addWidget(targetLangLabel)
        sideLayout.addWidget(self.targetLangSelector)
        
        # Add font family selector
        fontLabel = QLabel("Font Family:")
        self.fontSelector = QFontComboBox()
        self.fontSelector.currentFontChanged.connect(self.updateFontFamily)
        self.fontSelector.setCurrentFont(QFont("Comic Sans MS"))
        sideLayout.addWidget(fontLabel)
        sideLayout.addWidget(self.fontSelector)
        
        # Add border checkbox
        self.borderCheckbox = QCheckBox("Border")
        self.borderCheckbox.setChecked(False)
        sideLayout.addWidget(self.borderCheckbox)
        
        # Add stretch to push widgets to top
        sideLayout.addStretch()
        
        # Create toggle button (separate from the panel)
        toggleButton = QPushButton("◀")  # Left arrow
        toggleButton.setFixedWidth(20)
        toggleButton.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        toggleButton.clicked.connect(self.toggleRightSidePanel)
        
        return rightPanel, toggleButton

    def toggleLeftSidePanel(self):
        """Toggles the visibility of the left side panel."""
        if self.leftSidePanelVisible:
            self.leftSidePanel.hide()
            self.leftToggleButton.setText("▶") # Right arrow (to show panel)
        else:
            self.leftSidePanel.show()
            self.leftToggleButton.setText("◀") # Left arrow (to hide panel)
        self.leftSidePanelVisible = not self.leftSidePanelVisible

    def toggleRightSidePanel(self):
        """Toggles the visibility of the right side panel."""
        if self.rightSidePanelVisible:
            self.rightSidePanel.hide()
            self.rightToggleButton.setText("◀")  # Left arrow (to show panel)
        else:
            self.rightSidePanel.show()
            self.rightToggleButton.setText("▶")  # Right arrow (to hide panel)
        self.rightSidePanelVisible = not self.rightSidePanelVisible
    
    def keyPressEvent(self, event):
        """ Handle key press events for selection and other actions,
            it handles the shift key for selection and the escape key to cancel selection.
        """
        if event.key() == Qt.Key.Key_Shift:
            self.shiftPressed = True
        elif event.key() == Qt.Key.Key_Control:
            self.ctrlPressed = True
        if self.detectionToolSelected:
            if event.key() == Qt.Key.Key_Z and self.ctrlPressed:
                if len(self.allDisplayedItems) > 0:
                    item = self.allDisplayedItems.pop()
                    self.unDisplayBubble(item)
                    self.undisplayedItems.append(item)
            elif event.key() == Qt.Key.Key_Y and self.ctrlPressed:
                if len(self.undisplayedItems) > 0:
                    item = self.undisplayedItems.pop()
                    self.displayBubble(item)
                    self.allDisplayedItems.append(item)
            elif event.key() == Qt.Key.Key_Escape:
                self.isSelecting = False
                if self.selectionRect:
                    self.selectionRect.hide()
            elif event.key() == Qt.Key.Key_End:
                self.autoDetectAllBubbles()
        if self.textToolSelected:
            pass
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """ Handle key release events for selection and other actions,
            it handles the shift key for selection.
        """
        if event.key() == Qt.Key.Key_Shift:
            self.shiftPressed = False
        elif event.key() == Qt.Key.Key_Control:
            self.ctrlPressed = False
        super().keyReleaseEvent(event)

    def eventFilter(self, obj, event):
        """ Event filter to handle mouse events for selection and other actions. """
        if obj is self.view.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                return self.handleMousePress(event)
            elif event.type() == QEvent.Type.MouseMove:
                return self.handleMouseMove(event)
            elif event.type() == QEvent.Type.MouseButtonRelease:
                return self.handleMouseRelease(event)
        return super().eventFilter(obj, event)

    def handleMousePress(self, event):
        """ Handle mouse press events for selection and other actions. """
        if not self.pixmapItem or not self.shiftPressed:
            return False
        
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.view.mapToScene(event.pos())
            self.selectionStart = scene_pos
            self.isSelecting = True
            
            if not self.selectionRect:
                self.selectionRect = self.scene.addRect(
                    QRectF(scene_pos, scene_pos),
                    pen=QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
                )
                self.selectionRect.show()
                self.selectionRect.setZValue(1)
            else:
                self.selectionRect.setRect(QRectF(scene_pos, scene_pos))
                self.selectionRect.show()
                self.selectionRect.setZValue(1)
            return True
        return False

    def handleMouseMove(self, event):
        """ Handle mouse move events for selection and other actions. """
        if not self.isSelecting:
            return False
            
        currentPos = self.view.mapToScene(event.pos())
        rect = QRectF(self.selectionStart, currentPos).normalized()
        self.selectionRect.setRect(rect)
        pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
        self.selectionRect.setPen(pen)
        return True

    def handleMouseRelease(self, event):
        """ Handle mouse release events for selection and other actions. """
        if not self.isSelecting:
            return False
        
        if event.button() == Qt.MouseButton.LeftButton:
            self.isSelecting = False
            self.selectionRect.hide()
            
            rect = self.selectionRect.rect()
            x, y = rect.x(), rect.y()
            w, h = rect.width(), rect.height()
            verbosePrint(text=f"Selection: x={x}, y={y}, w={w}, h={h}", verboseLevel=self.verboseLevel, verboseThreshold=2)
            

            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
            self.selectionRect.setPen(pen)
            
            if self.detectionToolSelected:
                # Create mask for OCR
                mask = np.zeros((3, self.imageSize[1], self.imageSize[0]), dtype=np.uint8)
                mask[0, int(y):int(y+h), int(x):int(x+w)] = 255
                
                if not(self.ctrlPressed):
                    # With normal selection, we use prediction
                    self.startPrediction(imagePath=self.path[0], box=list((int(x), int(y), int(x+w), int(y+h))))
                else:
                    # With ctrl+selection, we directly use OCR on the mask
                    self.startOCR(self.path[0], mask)
                return True
            elif self.textToolSelected:
                pass
        return False

    def mouseDoubleClickEvent(self, event):
        """Handles double-click events to start prediction at the clicked point."""

        if not self.pixmapItem:
            return

        # Adjust event position for menubar and margins
        viewMargin = self.view.frameWidth()
        adjustedPos = event.pos()
        viewPos = self.view.mapFrom(self, adjustedPos)

        # Remove view frame margins
        viewPos.setX(viewPos.x() - viewMargin)
        viewPos.setY(viewPos.y() - viewMargin)

        # Convert to scene coordinates
        scene_pos = self.view.mapToScene(viewPos)

        # Get position relative to the pixmap item
        itemPos = self.pixmapItem.mapFromScene(scene_pos)
        points = [[int(itemPos.x()), int(itemPos.y())]]
        labels = [1]

        if self.detectionToolSelected:
            self.startPrediction(self.path[0], points, labels)
        elif self.textToolSelected:
            for bubble in self.allDisplayedItems:
                if bubble["bubble"].isPointInside(points[0]):
                    self.openTextEdit(bubble)
 
    def openTextEdit(self, bubbleObject):
        """Open a transparent text editor to edit the bubble text."""
        
        # Extract bubble information
        bubble = bubbleObject["bubble"]
        box = bubble.getBoundingBox()
        currentText = bubble.getBubbleText()
        
        # Remove existing text overlay temporarily
        if "textOverlay" in bubbleObject:
            self.scene.removeItem(bubbleObject["textOverlay"])
        
        # Calculate editor dimensions based on bubble size
        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]
        
        # Create a custom dialog for editing text
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Text")
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1A1A1A;
                border: 1px solid #FF00FF;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #2D2D2D;
                color: white;
                border: 1px solid #FF00FF;
                border-radius: 4px;
                selection-background-color: #FF00FF40;
            }
            QPushButton {
                background-color: #1E1E1E;
                color: white;
                border: 1px solid #FF00FF;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #2D2D2D;
                border: 1px solid #FF40FF;
            }
        """)
        
        # Create dialog layout
        layout = QVBoxLayout(dialog)
        
        # Create text edit widget
        textEdit = QTextEdit()
        textEdit.setText(currentText)
        textEdit.setMinimumSize(min(400, boxWidth + 50), min(300, boxHeight + 50))
        layout.addWidget(textEdit)
        
        # Add buttons
        buttonLayout = QHBoxLayout()
        saveButton = QPushButton("Save")
        cancelButton = QPushButton("Cancel")
        buttonLayout.addWidget(saveButton)
        buttonLayout.addWidget(cancelButton)
        layout.addLayout(buttonLayout)
        
        # Connect buttons
        def saveAndClose():
            # Update bubble text
            newText = textEdit.toPlainText()
            bubble.setBubbleText(newText)
            
            # Redraw text overlay
            self.writeOcrText(bubbleObject)
            dialog.accept()
        
        def cancelEdit():
            # Restore original text overlay
            self.writeOcrText(bubbleObject)
            dialog.reject()
        
        saveButton.clicked.connect(saveAndClose)
        cancelButton.clicked.connect(cancelEdit)
        
        # Position dialog near the bubble but ensure it's visible on screen
        dialogPos = self.mapToGlobal(self.view.mapFromScene(box[0], box[1]))
        dialog.move(dialogPos)
        
        # Create a glow effect for the dialog
        glow = QGraphicsDropShadowEffect(dialog)
        glow.setColor(QColor("#FF00FF"))
        glow.setOffset(0, 0)
        glow.setBlurRadius(15)
        dialog.setGraphicsEffect(glow)
        
        # Show dialog and handle result
        dialog.exec()
        
        # Make sure the JSON file is updated with the new text
        json_path = self.path[0].replace(".jpg", ".json").replace(".png", ".json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Find and update the bubble in the data
            index = bubbleIsNotInDict(bubble, data)
            if index != -1:
                data[index] = bubble.serialize(mask_storage='contour_delta')
                
                # Write back the updated data
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, cls=NumpyEncoder)
        except Exception as e:
            print(f"Error updating JSON: {e}")

    def toggleDetectionTool(self):
        """Switch to the detection tool."""
        self.detectionToolSelected = True
        self.textToolSelected = False
        
        # Update button appearance to show selection state
        self.detectionButton.setChecked(True)
        self.textEditButton.setChecked(False)
        
        # # Apply glow effect to selected tool
        # detect_glow = QGraphicsDropShadowEffect(self)
        # detect_glow.setColor(QColor("#FF00FF"))  # Neon pink
        # detect_glow.setOffset(0, 0)
        # detect_glow.setBlurRadius(15)
        # self.detectionButton.setGraphicsEffect(detect_glow)
        # self.textEditButton.setGraphicsEffect(None)

    def toggleTextTool(self):
        """Switch to the text editing tool."""
        self.detectionToolSelected = False
        self.textToolSelected = True
        
        # Update button appearance to show selection state
        self.detectionButton.setChecked(False)
        self.textEditButton.setChecked(True)
        
        # # Apply glow effect to selected tool
        # text_glow = QGraphicsDropShadowEffect(self)
        # text_glow.setColor(QColor("#FF00FF"))  # Neon pink
        # text_glow.setOffset(0, 0)
        # text_glow.setBlurRadius(15)
        # self.textEditButton.setGraphicsEffect(text_glow)
        # self.detectionButton.setGraphicsEffect(None)

        # self.detectionButton.setStyleSheet("background-color: #FFFFFF")
        # self.textEditButton.setStyleSheet("background-color: #A0A0A0")

    def updateFontFamily(self, font):
        """Updates the font family for text rendering."""
        self.currentFontFamily = font.family()

    def updateTargetLang(self, targetLang):

        self.targetLang = self.targetLangDict[targetLang]

    def openFunction(self, imagePath=None):
        """Opens an image file using a file dialog, remembering the last opened directory."""

        if not imagePath:
            # Get the last opened directory from settings, or use the current directory if not available
            lastDir = getattr(self, 'lastDir', '.')

            fileDialog = QFileDialog()
            fileDialog.setDirectory(lastDir)  # Set the initial directory

            self.path = fileDialog.getOpenFileName(
                self,
                'Open file',
                lastDir,  # Use lastDir as the directory
                "Image files (*.jpg *.gif *.png)"
            )
        else:
            self.path = (imagePath,)

        if self.path[0]:
            # Remember the directory for the next time
            self.lastDir = os.path.dirname(self.path[0])

            # Clear previous image and scene items
            self.scene.clear()
            self.allDisplayedItems = []
            self.pixmapItem = None
            self.selectionRect = None  # Reset selectionRect reference

            # Load and display new image
            pixmap = QPixmap(self.path[0])
            self.imageSize = pixmap.size().width(), pixmap.size().height()
            self.pixmapItem = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmapItem)
            self.undisplayedItems = []

            # Set scene rect to pixmap size for 1:1 scale
            self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            self.view.setSceneRect(0, 0, pixmap.width(), pixmap.height())

            # Disable scaling and fitting
            self.view.setTransform(QTransform())

            # Enable mouse tracking
            self.view.setMouseTracking(True)
            self.pixmapItem.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

            if os.path.exists(self.path[0].replace(".jpg", ".json").replace(".png", ".json")):
                with open(self.path[0].replace(".jpg", ".json").replace(".png", ".json"), "r", encoding="utf-8") as f:
                    data = json.load(f)
                for bubble in data:
                    bubble = OCRBubble.fromJSON(bubble)
                    whiteMask = createWhiteMaskPixmapFromNpMask(bubble.getWhiteMask(), border=self.borderCheckbox.isChecked())
                    bubbleObject = {"bubble":bubble, "whiteMask":whiteMask}
                    self.displayBubble(bubbleObject)
                    self.allDisplayedItems.append(bubbleObject)
                self.undisplayedItems = []

        else:
            verbosePrint(text="No file selected", verboseLevel=self.verboseLevel, verboseThreshold=1)

    def openDirectory(self):
        """Opens a directory using a file dialog, remembering the last opened directory.
           The directory is processed in sequence, with each image being processed in turn.
        Args:
            None
        Returns:
            None
        """

        # Get the last opened directory from settings, or use the current directory if not available
        lastDir = getattr(self, 'lastDir', '.')

        fileDialog = QFileDialog()
        fileDialog.setFileMode(QFileDialog.FileMode.Directory)
        fileDialog.setDirectory(lastDir)

        fileDialogResult = fileDialog.getExistingDirectory(self, 'Open directory', lastDir)
        if fileDialogResult:
            self.path = fileDialogResult
        else:
            return
        if self.path:
            # Remember the directory for the next time
            self.lastDir = self.path

            self.sequencedDirectoryProcessing(self.path)

    def saveFunction(self):
        """Saves the current scene as an image file in a directory called 'preprocessed' located in the parent directory of the image.
        Args:
            None
        Returns:
            None
        """

        # Determine the save path
        pathParts = self.path[0].replace("\\","/").split("/")
        filename = pathParts[-1]
        pathParts[-1] = filename
        pathParts.insert(-1, "preprocessed")
        saveDir = "/".join(pathParts[:-1])
        if not os.path.exists(saveDir):
            os.makedirs(saveDir, exist_ok=True)
        save_path = "/".join(pathParts)

        # Create a QImage of the scene
        sceneRect = self.scene.itemsBoundingRect()
        image = QImage(sceneRect.size().toSize(), QImage.Format.Format_ARGB32)
        image.fill(QColor(255, 255, 255, 255))  # Fill with white background

        # Render the scene onto the image
        painter = QPainter(image)
        self.scene.render(painter, QRectF(image.rect()), sceneRect)
        painter.end()

        # Save the image
        if image.save(save_path):
            print(f"Scene saved to {save_path}")
        else:
            print(f"Error saving scene to {save_path}")

    def startPrediction(self, imagePath, points=None, labels=None, box=None):
        """ Start a thread to predict the mask for the given image and points.
            There are two modes of operation:
            1. If points and labels are provided, predict the mask for the region of interest.
            2. If box is provided, predict the mask for the bounding box.
        Args:
            imagePath (str): Path to the image file.
            points (List[List[int]]): List of points for the region of interest.
            labels (List[int]): List of labels for the points.
            box (List[int]): Bounding box for the region of interest.
        Returns:
            None"""
        thread = FullProcessingThread(self, imagePath, points, labels, box, verboseLevel=self.verboseLevel)
        thread.finished.connect(self.addBubbleObjectToScene)
        thread.finished.connect(thread.deleteLater)  # Ensure the thread is deleted properly
        self.activeThread.append(thread)
        thread.start()

    def startOCR(self, imagePath, mask):
        """ Start a thread to perform OCR on the given mask.
        Args:
            imagePath (str): Path to the image file.
            mask (np.ndarray): Binary mask image with 255 for the region of interest.
        Returns:
            None
        """
        # Start the translation thread with the thread_id and mask
        TransThread = OnlyOCRProcessingThread(self, imagePath, mask, verboseLevel=self.verboseLevel)
        TransThread.finished.connect(TransThread.deleteLater)
        TransThread.finished.connect(self.addBubbleObjectToScene)
        self.activeThread.append(TransThread)
        TransThread.start()

    def addBubbleObjectToScene(self):
        """Add a layer to the scene with the mask"""
        while not(self.predictionOutputQueue.empty()):
            output = self.predictionOutputQueue.get()
           
            

            box = output.getBoundingBox()
            bubbleObject = {"bubble":output}
            whiteMask_array = output.getWhiteMask()
            if whiteMask_array is not None:
                # Create pixmap from existing mask
                whiteMask = createWhiteMaskPixmapFromNpMask(whiteMask_array, border=self.borderCheckbox.isChecked())
            else:
                # Create a NumPy mask from the bounding box
                mask = np.zeros((3, self.imageSize[1], self.imageSize[0]), dtype=np.uint8)
                x_min, y_min, x_max, y_max = box
                # Set the mask values to 255 within the bounding box region
                mask[0, int(y_min):int(y_max), int(x_min):int(x_max)] = 255
            
                # Store the mask in the bubble object
                output.setWhiteMask(mask)

                whiteMask = QGraphicsRectItem(box[0], box[1], box[2] - box[0], box[3] - box[1])
                whiteMask.setBrush(QColor(255, 255, 255, 255))
                penColor = Qt.GlobalColor.black if self.borderCheckbox.isChecked() else Qt.GlobalColor.white
                whiteMask.setPen(QPen(penColor, 2, Qt.PenStyle.SolidLine))

            bubbleObject["whiteMask"] = whiteMask
            
            self.displayBubble(bubbleObject)
            self.allDisplayedItems.append(bubbleObject)
            self.predictionOutputQueue.task_done()
            
    def writeOcrText(self, bubbleObject):
        """Write OCR text to the scene"""
                
        text = bubbleObject["bubble"].getBubbleText()
        box = bubbleObject["bubble"].getBoundingBox()
        # Create transparent pixmap
        overlay = QPixmap(self.imageSize[0], self.imageSize[1])
        overlay.fill(Qt.GlobalColor.transparent)
        
        # Create painter for the pixmap
        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Calculate box dimensions
        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]
        
        # Configure font
        font = painter.font()
        font.setFamily(getattr(self, 'currentFontFamily', "Comic Sans MS"))
        
        optimalSize, optimalLines, padding = computeOptimalFontSize(font, text, boxWidth, boxHeight, painter)
        
        # Set final font size
        font.setPointSize(optimalSize)
        painter.setFont(font)
        
        # Configure text color
        painter.setPen(QColor(0, 0, 0, 255))  # Black color with full opacity
        
        # Get text metrics with final font
        metrics = painter.fontMetrics()
        lineHeight = metrics.height()
        
        # If we didn't find optimal lines (rare case), recalculate with final font size
        if not optimalLines:
            words = text.split()
            optimalLines = []
            currentLine = ""
            usable_width = boxWidth - (padding * 2)
            
            for word in words:
                testLine = currentLine + (" " if currentLine else "") + word
                if metrics.horizontalAdvance(testLine) <= usable_width:
                    currentLine = testLine
                else:
                    if currentLine:
                        optimalLines.append(currentLine)
                    currentLine = word
            
            if currentLine:
                optimalLines.append(currentLine)
        
        # Calculate vertical centering
        totalTextHeight = len(optimalLines) * lineHeight
        yStart = box[1] + (boxHeight - totalTextHeight) // 2
        
        # Draw text lines
        y = yStart
        for line in optimalLines:
            # Center text horizontally
            lineWidth = metrics.horizontalAdvance(line)
            x = box[0] + (boxWidth - lineWidth) // 2
            
            painter.drawText(x, y + metrics.ascent(), line)
            y += lineHeight
        
        painter.end()
        
        # Add overlay to scene
        overlayItem = QGraphicsPixmapItem(overlay)
        self.scene.addItem(overlayItem)
        bubbleObject["textOverlay"] = overlayItem
        self.undisplayedItems = []     
    
    def predictBoundingsBoxes(self):
        """Predict bounding boxes for all speech bubbles in the image."""
        imagePath = self.path[0]
        result = self.yolo.predict(imagePath)[0]

        
        filtered_boxes = []
        for i in range(len(result.boxes)):
            if result.boxes.conf[i].item() > 0.7:
                filtered_boxes.append(result.boxes.xyxy.tolist()[i])

        return filtered_boxes

    def autoDetectAllBubbles(self):
        """Automatically detect all speech bubbles in the image."""
        boxes = self.predictBoundingsBoxes()
        for box in boxes:
            self.startPrediction(imagePath=self.path[0], box=box)
        
    def sequencedDirectoryProcessing(self, directory):
        """Process all images in a directory in sequence, waiting for each to complete."""
        verbosePrint(text="Processing directory...", verboseLevel=self.verboseLevel, verboseThreshold=1)
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        print(image_files)
        
        for image_file in image_files:
            imagePath = os.path.join(directory, image_file)
            verbosePrint(text=f"Processing {image_file}...", verboseLevel=self.verboseLevel, verboseThreshold=1)
            
            # Open the current image
            self.openFunction(imagePath)
            
            # Set up tracking for active threads
            self.active_threads = []
            
            # Start bubble detection
            boxes = self.predictBoundingsBoxes()
            if not boxes:
                verbosePrint(text="No speech bubbles detected in image.", verboseLevel=self.verboseLevel, verboseThreshold=1)
                continue
                
            # Create event loop to wait for completion
            loop = QEventLoop()
            processed_count = 0
            total_boxes = len(boxes)
            
            def check_completion():
                nonlocal processed_count
                processed_count += 1
                if processed_count >= total_boxes:
                    loop.quit()
            
            # Start prediction threads for each box
            for box in boxes:
                thread = FullProcessingThread(self, imagePath, box=box, verboseLevel=self.verboseLevel)
                thread.finished.connect(self.addBubbleObjectToScene)
                thread.finished.connect(check_completion)
                thread.finished.connect(thread.deleteLater)
                self.active_threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            if total_boxes > 0:
                verbosePrint(text="Waiting for threads to complete...", verboseLevel=self.verboseLevel, verboseThreshold=1)
                loop.exec()
            
            # Save the processed image
            self.saveFunction()
            
            # Clear queues and state for next image
            self.predictionOutputQueue = queue.Queue()
            self.allDisplayedItems = []
            
            self.active_threads = []
            
            verbosePrint(text=f"Processing of {image_file} complete.", verboseLevel=self.verboseLevel, verboseThreshold=1)
            
        verbosePrint(text="Directory processing complete.", verboseLevel=self.verboseLevel, verboseThreshold=1)

    def displayBubble(self, bubbleObject):
        """Display a bubble object in the scene."""
        whiteMask = bubbleObject["whiteMask"]
        self.scene.addItem(whiteMask)
        if "textOverlay" in bubbleObject.keys():
            self.scene.addItem(bubbleObject["textOverlay"])
        else:
            self.writeOcrText(bubbleObject)


        json_path = self.path[0].replace(".jpg", ".json").replace(".png", ".json")
        
        # First try to read existing data
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        except FileNotFoundError:
            data = []
        
        # Add new bubble data
        if bubbleIsNotInDict(bubbleObject["bubble"], data) == -1:
            data.append(bubbleObject["bubble"].serialize(mask_storage='contour_delta'))  # Use RLE for efficient storage
        
        # Now write back the updated data
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyEncoder)

    def unDisplayBubble(self, bubbleObject):
        """Remove a bubble object from the scene."""
        whiteMask = bubbleObject["whiteMask"]
        textOverlay = bubbleObject["textOverlay"]
        self.scene.removeItem(whiteMask)
        self.scene.removeItem(textOverlay)

        json_path = self.path[0].replace(".jpg", ".json").replace(".png", ".json")

        # First try to read existing data
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        except FileNotFoundError:
            data = []

        # Remove bubble data
        index = bubbleIsNotInDict(bubbleObject["bubble"], data)
        if index != -1:
            data.pop(index)
        
        # Now write back the updated data
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyEncoder)
            
            
    
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.addMainMenu()
    window.show()
    sys.exit(app.exec())