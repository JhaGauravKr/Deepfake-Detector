# 🕵️ Deepfake Detector Web Application

A robust Django-based web application designed to detect deepfake videos using a combination of ResNeXt50 for spatial feature extraction and an LSTM network for temporal analysis. Users can upload video files, and the application will analyze them to predict whether the content is "REAL" or "FAKE," along with a confidence score.

## ✨ Features

* **Video Upload:** Intuitive web interface for uploading video files.
* **Video Preprocessing:** Extracts frames from uploaded videos.
* **Face Detection & Cropping:** Utilizes `face_recognition` to identify and crop facial regions from video frames for focused analysis.
* **Deep Learning Model:**
    * **ResNeXt50 Backbone:** Extracts powerful spatial features from individual frames.
    * **LSTM Network:** Analyzes temporal dependencies across a sequence of frames, crucial for detecting dynamic inconsistencies in deepfakes.
* **Prediction & Confidence Score:** Provides a clear "REAL" or "FAKE" prediction with a confidence percentage.
* **Dynamic Model Loading:** Automatically selects the most accurate pre-trained model based on the user-specified sequence length from available models.
* **Responsive UI:** A clean, modern, and responsive user interface built with HTML and CSS for a smooth experience across devices.
* **Error Handling:** Robust error pages for common issues like file upload limits, missing models, or processing failures.

## 🛠️ Technologies Used

* **Backend:**
    * Python 3.x
    * Django (Web Framework)
    * PyTorch (Deep Learning Framework)
    * Torchvision (for ResNeXt50 model and transforms)
* **Computer Vision & ML Libraries:**
    * OpenCV (`opencv-python`)
    * Face-Recognition
    * NumPy
    * Pillow (PIL Fork)
    * Scikit-image (for some image ops, though mainly OpenCV/PIL used)
* **Frontend:**
    * HTML5
    * CSS3
    * Google Fonts (Inter)
    * Font Awesome (for icons, optional)
* **Database:** SQLite3 (development)

## 🚀 Installation & Setup

Follow these steps to get the project up and running on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Steps

1. **Clone the Repository:**
    Open your terminal or command prompt and navigate to the directory where you want to save the project. Then, run the following command:

    ```bash
    git clone [https://github.com/JhaGauravKr/Deepfake-Detector.git](https://github.com/JhaGauravKr/Deepfake-Detector.git)
    cd Deepfake-Detector # Navigate into the cloned project directory
    ```
    *This command will create a folder named `Deepfake-Detector` in your current directory and download all project files into it.*

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For CUDA/GPU support with PyTorch, you might need to follow specific installation instructions from [PyTorch.org](https://pytorch.org/get-started/locally/) based on your system configuration and CUDA version. The `requirements.txt` installs the CPU version by default.*

4.  **Prepare Trained Models:**
    * Create a `models` directory in your project root (same level as `manage.py`).
        ```bash
        mkdir models
        ```
    * Place your pre-trained PyTorch model `.pt` files inside this `models/` directory.
    # Download Models
      https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing

    * **Important:** Model filenames must follow the convention:
        `model_ACCURACY_acc_SEQUENCE_LENGTH_frames_FF_data.pt`
        (e.g., `model_87_acc_60_frames_FF_data.pt`, `model_92_acc_100_frames_FF_data.pt`). The accuracy can be an integer or float. Ensure you have models for the sequence lengths you intend to support (e.g., 10, 20, 40, 60, 100 frames).

5.  **Run Database Migrations:**
    ```bash
    python manage.py migrate
    ```
    *(This creates the necessary database tables, including for sessions, authentication, etc.)*

6.  **Start the Django Development Server:**
    ```bash
    python manage.py runserver
    ```

## 🌐 Usage

1.  Open your web browser and navigate to `http://127.0.0.1:8000/`.
2.  On the homepage, use the "Upload Video File" field to select a video (e.g., `.mp4`, `.gif`, `.webm`).
3.  Enter the desired "Sequence Length" (e.g., `60`) – this determines how many frames will be analyzed.
4.  Click "Analyze Video."
5.  The application will process the video, extract faces, run the deepfake detection model, and display the prediction (REAL/FAKE) along with a confidence score. You'll also see preview images of preprocessed frames and detected faces.



## 3. Results

| Model Name | No of videos | No of Frames | Accuracy |
|------------|--------------|--------------|----------|
|model_84_acc_10_frames_FF_data.pt |6000 |10 |84.21461|
|model_87_acc_20_frames_FF_data.pt | 6000 |20 |87.79160|
|model_89_acc_40_frames_FF_data.pt | 6000| 40 |89.34681|
|model_90_acc_60_frames_FF_data.pt | 6000| 60 |90.59097 |
|model_91_acc_80_frames_FF_data.pt | 6000 | 80 | 91.49818 |
|model_93_acc_100_frames_FF_data.pt| 6000 | 100 | 93.58794|


Always use maximum frames for better result.

## 🏗️ Model Architecture

### 🔹 1. CNN Backbone (ResNet/ResNeXt)
- Input: RGB video frames `(3 channels)`
- Initial convolution: `Conv(3 → 64, kernel=7x7)`
- Batch Normalization + Activation
- Residual bottleneck blocks with channel progression:
  ```
  64 → 128 → 256 → 512 → 1024 → 2048
  ```
- Global average pooling per frame
- **Output per frame:** `2048-dimensional feature vector`

### 🔹 2. LSTM Layer
- Input size: **2048** (CNN feature per frame)
- Hidden size: **2048**
- Learns temporal dynamics such as:
  - Eye blinking patterns
  - Lip-sync smoothness
  - Head motion
  - Temporal coherence

### 🔹 3. Linear Classifier
- Fully connected layer: `2048 → 2`
- Output: `[real, fake]` probability distribution

---


---

## 📊 Block Diagram
![Block Diagram](ModelDiagram.png)

---

## 📂 Dataset
- Input: RGB video clips
- Preprocessing:
  - Extract frames
  - Resize and normalize
  - Sequence batching for LSTM
- Labels: `real`, `fake`

---

## ⚙️ Training Details
- Loss: Cross-Entropy
- Optimizer: Adam / SGD
- Framework: PyTorch
- Checkpoints stored as `.pt` files
- Example parameter sizes:
  - `lstm.weight_ih_l0 → (8192, 2048)`
  - `linear1.weight → (2, 2048)`


---

## 🧠 Key Insights
- **CNN** captures **spatial inconsistencies** (blurring, artifacts, texture issues).
- **LSTM** captures **temporal inconsistencies** (unnatural motion, blink irregularities).
- **Hybrid design** → stronger performance than CNN-only models.

---

## 📌 Applications
- Social media content verification
- Fake news prevention
- Video authentication in forensics

---


## 📄 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT). You can create a `LICENSE` file in your root with the content.


