PANOPTIC SEGMENTATION

What is Panoptic Segmentation?
------------------------------
Panoptic Segmentation combines:

1. Semantic Segmentation
   - Labels every pixel according to its class.
   - Background elements such as roads, sky, buildings, etc. are identified.
   - Individual objects of the same class are NOT separated.

2. Instance Segmentation
   - Separates individual objects of the same class.
   - Example:
       Person 1
       Person 2
       Person 3
   - Focuses on object instances rather than background.

Panoptic Segmentation = Semantic Segmentation + Instance Segmentation

------------------------------------------------------------

CNN (Convolutional Neural Network)
----------------------------------
CNN is a deep learning architecture used for analyzing and processing visual data such as images and videos.

ResNet50 is a CNN model.

------------------------------------------------------------

PROJECT PIPELINE
----------------

1. COCO Dataset
---------------
COCO (Common Objects in Context) is a large dataset containing:

- Images
- Object Labels
- Segmentation Masks

It contains 80 object categories such as:

- Person
- Car
- Dog
- Bicycle
- Chair

Detectron2 models are often pre-trained on the COCO dataset.

------------------------------------------------------------

2. ResNet50
-----------
ResNet50 acts as the backbone network.

Its job is to extract important features from the input image such as:

- Edges
- Shapes
- Textures
- Patterns
- Size information

In simple terms, ResNet50 converts the image into meaningful feature maps that the model can understand.

------------------------------------------------------------

3. FPN (Feature Pyramid Network)
--------------------------------
Objects in images can appear at different sizes.

FPN creates feature maps at multiple scales, helping the model detect:

- Small objects
- Medium objects
- Large objects

Without FPN, small objects may be missed.

------------------------------------------------------------

4. ResNet50 + FPN
-----------------
The output from ResNet50 is enhanced by FPN.

These combined feature maps are then provided as input to Detectron2.

------------------------------------------------------------

5. Detectron2
-------------
Detectron2 is the complete computer vision framework.

It uses:

- ResNet50 as the backbone
- FPN as the neck
- COCO dataset knowledge learned during training

ResNet50 extracts features from the image.

FPN improves these features at multiple scales.

Detectron2 uses these features to:

- Classify objects
- Locate objects
- Segment objects

------------------------------------------------------------

WORKFLOW
--------

Input Image
      |
      v
ResNet50
(Feature Extraction)
      |
      v
FPN
(Multi-scale Feature Maps)
      |
      v
Detectron2
(Uses COCO-trained knowledge)
      |
      v
Panoptic Segmentation Output

------------------------------------------------------------
