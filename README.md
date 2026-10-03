# 🔬 Automatically Counting and Analyzing Cells in Microscopy Images

An image-processing pipeline for **automatically detecting, counting, and visualizing cells in microscopy images** using Python and OpenCV.

The project applies classical computer-vision techniques—including grayscale conversion, Gaussian smoothing, Otsu thresholding, morphological processing, and contour detection—to identify individual cell-like objects and display their locations with bounding boxes.

---

## 🧬 Project Overview

Manual cell counting from microscopy images can be time-consuming and difficult to reproduce consistently.

This project demonstrates how image-processing techniques can automate a basic cell-counting workflow:

**Microscopy Image → Preprocessing → Thresholding → Morphology → Contour Detection → Cell Filtering → Counting → Visualization**

The current implementation is designed as a lightweight foundation that can be extended toward more advanced microscopy image-analysis workflows.

---

## ✨ Features

* 🖼️ Load microscopy images using OpenCV
* ⚫ Convert images to grayscale
* 🌫️ Reduce image noise using Gaussian blur
* 🎯 Automatically segment objects using Otsu thresholding
* 🧹 Improve segmentation using morphological operations
* 🔍 Detect cell-like objects using contour detection
* 📏 Filter small contours using an area threshold
* 🔢 Automatically count detected cells
* 📦 Draw bounding boxes around detected cells
* 📊 Visualize the final results using Matplotlib

---

## 🛠️ Technologies Used

| Technology     | Purpose                              |
| -------------- | ------------------------------------ |
| **Python**     | Core programming language            |
| **OpenCV**     | Image processing and computer vision |
| **NumPy**      | Numerical array operations           |
| **Matplotlib** | Visualization                        |

---

## 📂 Project Structure

```text
Automatically-counting-and-analyzing-cells-in-microscopy-images/
│
├── README.md
├── script.py
└── microscopy_image.jpg
```

> `microscopy_image.jpg` is the example input expected by the current script. You can replace it with your own microscopy image.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Bioinformatician-dev/Automatically-counting-and-analyzing-cells-in-microscopy-images.git
```

Navigate into the project:

```bash
cd Automatically-counting-and-analyzing-cells-in-microscopy-images
```

Install the required Python packages:

```bash
pip install opencv-python numpy matplotlib
```

---

## 🚀 Usage

Place your microscopy image in the project directory and name it:

```text
microscopy_image.jpg
```

Then run:

```bash
python script.py
```

The program will process the image and display the detected cells with bounding boxes.

---

## 🔬 Methodology

### 1. Load the Image

The microscopy image is loaded using OpenCV.

```python
image = cv2.imread(image_path)
```

The image is then converted from BGR to grayscale:

```python
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 2. Gaussian Blur

Gaussian filtering is applied to reduce noise and smooth the image:

```python
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
```

### 3. Otsu Thresholding

Otsu's thresholding automatically determines a threshold for separating foreground objects from the background:

```python
_, binary_image = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

### 4. Morphological Processing

Morphological closing is applied using an elliptical kernel to help connect small gaps in segmented objects:

```python
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

morph_image = cv2.morphologyEx(
    binary_image,
    cv2.MORPH_CLOSE,
    kernel
)
```

### 5. Contour Detection

External contours are detected from the processed image:

```python
contours, _ = cv2.findContours(
    morph_image,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

### 6. Cell Filtering

Very small contours are removed using an area threshold:

```python
if cv2.contourArea(contour) > 100:
```

This helps reduce detections caused by small artifacts or noise.

### 7. Visualization

Bounding boxes are drawn around detected objects and the total count is reported:

```text
Total number of cells detected: N
```

The current implementation then displays the annotated microscopy image using Matplotlib.

---

## 📊 Output

The pipeline produces:

1. **Total detected cell count**
2. **Annotated microscopy image**
3. **Bounding boxes around detected objects**

Example workflow:

```text
Input Microscopy Image
          ↓
     Grayscale
          ↓
    Gaussian Blur
          ↓
  Otsu Thresholding
          ↓
Morphological Closing
          ↓
 Contour Detection
          ↓
  Area Filtering
          ↓
    Cell Counting
          ↓
 Bounding Box Visualization
```

---

## 🧪 Example Result

The output image highlights detected cell-like objects with bounding boxes, making it easier to visually inspect which objects were included in the automated count.

> For best results, use microscopy images with relatively clear contrast between cells and background.

---

## ⚠️ Limitations

This project currently uses **classical image-processing techniques**, so performance can depend strongly on image quality and experimental conditions.

Potential challenges include:

* Overlapping cells
* Low contrast between cells and background
* Uneven illumination
* Highly variable cell morphology
* Background artifacts
* Connected cells being detected as a single object
* Fragmented cells producing multiple contours
* Images requiring different segmentation thresholds

Therefore, the detected count should be considered an automated image-processing estimate rather than a universally accurate biological measurement.

---

## 🚀 Future Improvements

The project can be expanded into a more comprehensive microscopy-analysis pipeline.

### 🔹 Advanced Cell Segmentation

Implement:

* Watershed segmentation
* Distance transforms
* Adaptive thresholding
* Connected-component analysis

### 🔹 Quantitative Cell Analysis

Extract features such as:

* Cell area
* Perimeter
* Diameter
* Circularity
* Aspect ratio
* Solidity
* Bounding-box dimensions

### 🔹 Cell Classification

Extend the pipeline to classify cells according to:

* Morphology
* Size
* Shape
* Fluorescence intensity
* Cell type

### 🔹 Deep Learning

A future version could use modern segmentation or detection models such as:

* U-Net
* Cellpose
* StarDist
* YOLO-based detection/segmentation
* Mask R-CNN

### 🔹 Interactive Application

The workflow could be converted into an interactive application using:

* Streamlit
* Gradio
* Jupyter
* Hugging Face Spaces

Users could upload a microscopy image and receive:

```text
Cell Count
Cell Measurements
Annotated Image
CSV Results
```

---

## 📈 Potential Applications

This type of automated image analysis can serve as a foundation for:

* Cell counting
* Microscopy image analysis
* Cell morphology analysis
* Biological image processing
* High-throughput screening
* Quantitative cell biology
* Computer vision in biomedical research

---

## 🎯 Learning Objectives

This project demonstrates practical applications of:

* Python programming
* OpenCV
* Digital image processing
* Image segmentation
* Thresholding
* Morphological operations
* Contour detection
* Object detection
* Biomedical image analysis

---

## 👩‍🔬 Project Context

This repository demonstrates the application of **computational methods and computer vision to biological imaging**, bridging biology with programming and quantitative image analysis.

It can serve as a starting point for developing more advanced **AI-assisted microscopy and biomedical image-analysis pipelines**.

---

## 📜 License

This project is intended for educational and research purposes.

---

## ⭐ Acknowledgment

Built with:

**Python • OpenCV • NumPy • Matplotlib**

If you find this project useful, consider giving the repository a ⭐ on GitHub.
