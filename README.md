# BubbleTron

BubbleTron is an application designed to translate Japanese manga, offering both fully and partially automated workflows. It leverages several powerful tools:

* **Dialog Bubble Detection:** A fine-tuned YOLOv8 model (thanks to Denis Topallaj - Detopall) is used to detect dialog bubbles within manga images.
* **Bubble Segmentation:** The Segment Anything Model (SAM2) from Meta is then employed to precisely segment the detected bubbles.
* **Optical Character Recognition (OCR):** Manga-OCR (thanks to Maciej Budyś - kha-white) extracts the Japanese text from the segmented bubbles.
* **Translation:** The DeepL translation API is used to translate the extracted text. (Note: A DeepL API account, even a free one, is required. You will need to provide your API token.)

## Future Improvements

* **Text Editing:** Implement functionality to allow users to edit the translated text.
* **Data Serialization:** Serialize bubble data and store it in a separate file for later use or modification.
* **Background Reconstruction:** Explore the implementation of diffusion-based background reconstruction for areas where text has been removed (non-bubbled text).

## Version Notes

**v1.0.0 - First Release:**

* Directory processing: Process all image files within a specified directory.
* Single file processing: Process a single manga image file.
* Manual selection: Enable manual selection of non-bubbled text areas for translation.

## Acknowledgments

* Denis Topallaj - Detopall: [https://github.com/Detopall/manga-translator](https://github.com/Detopall/manga-translator)
* Maciej Budyś - kha-white: [https://github.com/kha-white/manga-ocr](https://github.com/kha-white/manga-ocr)
