import cv2
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo


# -----------------------------
# Create output folder
# -----------------------------
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)


# -----------------------------
# Load Panoptic Model
# -----------------------------
cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    )
)

cfg.MODEL.WEIGHTS = "model_final_c10459.pkl"
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


# -----------------------------
# Confidence threshold
# -----------------------------
CONF_THRESHOLD = 0.6


# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Webcam not detected")
    exit()

print("\nPress C → Capture image")
print("Press Q → Quit\n")

image_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF


    # -----------------------------
    # Capture Image
    # -----------------------------
    if key == ord("c"):

        print("\nProcessing image...\n")

        outputs = predictor(frame)

        panoptic_seg, segments_info = outputs["panoptic_seg"]

        objects = defaultdict(int)


        # -----------------------------
        # Extract objects
        # -----------------------------
        for seg in segments_info:

            # Skip very low confidence detections
            if "score" in seg and seg["score"] < CONF_THRESHOLD:
                continue

            class_id = seg["category_id"]

            try:
                if seg["isthing"]:
                    if class_id < len(metadata.thing_classes):
                        class_name = metadata.thing_classes[class_id]
                    else:
                        continue
                else:
                    if class_id < len(metadata.stuff_classes):
                        class_name = metadata.stuff_classes[class_id]
                    else:
                        continue

                objects[class_name] += 1

            except:
                continue


        # -----------------------------
        # Structured Output
        # -----------------------------
        print("Detected Objects\n")

        for obj, count in objects.items():

            name = obj.capitalize()

            if count == 1:
                print(name)

            else:
                print(name + "s")

                for i in range(count):
                    print("  " + name + " " + str(i+1))

        print("\n")


        # -----------------------------
        # Visualization
        # -----------------------------
        v = Visualizer(frame[:, :, ::-1], metadata, scale=1.2)

        out = v.draw_panoptic_seg_predictions(
            panoptic_seg.to("cpu"),
            segments_info
        )

        result = out.get_image()[:, :, ::-1]


        # -----------------------------
        # Save segmented image
        # -----------------------------
        image_count += 1

        filename = os.path.join(
            output_folder,
            f"panoptic_{image_count}.png"
        )

        cv2.imwrite(filename, result)

        print("Saved:", filename)


        cv2.imshow("Panoptic Segmentation", result)


    # -----------------------------
    # Quit program
    # -----------------------------
    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
