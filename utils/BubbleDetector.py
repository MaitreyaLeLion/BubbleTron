from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.visualization import show_masks
from PIL import Image
import numpy as np
import threading
import queue

class BubbleDetector:
    def __init__(self, buffer_size=5):
        self.model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml","models/SAM2/models/sam2.1_hiera_large.pt",device="cuda")
        self.image_predictor = None
        self.lock = threading.Lock()
        self.task_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.worker_thread.start()

    def _process_tasks(self):
        while True:
            image_path, points, labels, box = self.task_queue.get()
            print(f"Processing task for image {image_path}")
            try:
                masks, scores = self._predict(image_path, points, labels, box)
                maxScore = max(scores)
                masks = [mask for mask, score in zip(masks, scores) if score == maxScore]
                scores = [score for score in scores if score == maxScore]
                masks = np.array(masks)
                scores = np.array(scores)
                print(f"Predicted {len(masks)} masks with scores {scores}")
                self.result_queue.put((masks, scores))
            except Exception as e:
                print(f"Error processing task: {e}")
                self.result_queue.put(None)  # Indicate an error
            finally:
                self.task_queue.task_done()

    def _predict(self, image_path, points, labels, box):
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        
        image = Image.open(image_path)
        image = np.array(image.convert('RGB'))
        
        with self.lock:
            self.image_predictor = SAM2ImagePredictor(self.model)
            self.image_predictor.set_image(image)
            masks, scores, logits = self.image_predictor.predict(point_coords=input_points, point_labels=input_labels, box=box, multimask_output=True)
        return masks, scores

    def predict(self, image_path, points=None, labels=None, box=None, timeout=None):
        try:
            self.task_queue.put((image_path, points, labels, box), timeout=timeout)
            return self.result_queue.get(timeout=timeout)
        except queue.Full:
            print("Task queue is full")
            return None
        except queue.Empty:
            print("Result queue is empty (timeout)")
            return None

    def close(self):
        # Signal the worker thread to exit (optional)
        # In this example, the worker thread is a daemon, so it will exit when the main thread exits.
        pass

if __name__ == '__main__':
    bd = BubbleDetector()
    points = [[148,109]]
    labels = [1]
    output_masks = bd.predict("C:/IA/BubbleTron/Sam_pointer/5.jpg", points, labels)
    if output_masks:
        masks, scores = output_masks
        # Assuming show_masks can handle numpy arrays directly
        show_masks(masks)
    else:
        print("Prediction failed")
    bd.close()