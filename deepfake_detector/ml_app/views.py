# ml_app/views.py

from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm # Ensure this is correctly imported

# --- Global Configurations & Templates ---
index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"
error_404_template_name = '404.html'
cuda_full_template_name = 'cuda_full.html'

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1) # Corrected: Softmax needs a dimension

# Corrected: inv_normalize for visualization purposes
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

# Determine device for PyTorch operations
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data transformation pipeline for input frames
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- Model Definition ---
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        # Load pre-trained ResNeXt50 model with default weights
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        # Extract features by removing the classification layers
        self.model = nn.Sequential(*list(model.children())[:-2])
        # LSTM layer to process sequences of features
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU() # Activation function (not used in forward method currently)
        self.dp = nn.Dropout(0.4) # Dropout for regularization
        # Linear layer for final classification
        self.linear1 = nn.Linear(hidden_dim, num_classes) # Changed latent_dim to hidden_dim here
        self.avgpool = nn.AdaptiveAvgPool2d(1) # Adaptive pooling to get 1x1 feature map

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        # Reshape for CNN processing: (batch_size * seq_length, c, h, w)
        x = x.view(batch_size * seq_length, c, h, w)
        # Get feature maps from the CNN backbone
        fmap = self.model(x)
        # Apply adaptive average pooling
        x = self.avgpool(fmap)
        # Reshape for LSTM: (batch_size, seq_length, features_dim)
        x = x.view(batch_size, seq_length, -1) # -1 infers the feature dimension (2048)
        # Pass through LSTM
        x_lstm, _ = self.lstm(x, None) # _ receives (h_n, c_n) which we don't need directly
        # Take the output of the last time step from LSTM and apply linear layer
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# --- Dataset Definition ---
class ValidationDataset(Dataset): # Renamed for clarity (Dataset is already in torchvision)
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        
        # Extract and process frames from the video
        for i, frame in enumerate(self.frame_extract(video_path)):
            if frame is None: # Handle cases where cv2.read() might return None
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_frame)
            
            processed_frame = rgb_frame # Default to full frame if no face or error
            if faces:
                top, right, bottom, left = faces[0]
                # Apply padding safely within image bounds
                h, w, _ = rgb_frame.shape
                padding = 40 # Consistent padding
                top = max(0, top - padding)
                left = max(0, left - padding)
                bottom = min(h, bottom + padding)
                right = min(w, right + padding)
                
                cropped_face = rgb_frame[top:bottom, left:right]
                if cropped_face.size > 0: # Ensure cropped image is not empty
                    processed_frame = cropped_face
                else:
                    print(f"Warning: Cropped face for frame {idx}-{i} is empty after bounds check. Using full frame.")

            frames.append(self.transform(processed_frame))
            if len(frames) == self.count:
                break
        
        # If fewer frames are extracted than self.count, duplicate the last frame or add black frames
        while len(frames) < self.count:
            if frames:
                # Duplicate the last successfully processed frame
                frames.append(frames[-1])
            else:
                # If no frames could be extracted at all, create a black image tensor
                dummy_frame_np = np.zeros((im_size, im_size, 3), dtype=np.uint8)
                frames.append(self.transform(dummy_frame_np))

        frames = torch.stack(frames)
        return frames.unsqueeze(0) # Add batch dimension for DataLoader (batch_size=1)
    
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        if not vidObj.isOpened():
            print(f"Error: Could not open video file for frame extraction: {path}")
            return # Exit generator if video cannot be opened

        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image
        vidObj.release()

# --- Image Utility Functions ---
def im_convert(tensor, video_file_name):
    """ Converts a PyTorch tensor to a NumPy image for display/saving. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze() # Remove batch and channel dimensions if present
    image = inv_normalize(image) # Denormalize the image
    image = image.numpy()
    image = image.transpose(1, 2, 0) # Change from (C, H, W) to (H, W, C)
    image = image.clip(0, 1) # Clip values to [0, 1] for proper display
    return (image * 255).astype(np.uint8) # Convert to 0-255 range and uint8 for OpenCV/PIL

def im_plot(tensor): # This function is not used in your current views, but kept for completeness
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = image * [0.22803, 0.22145, 0.216989] + [0.43216, 0.394666, 0.37645]
    image = image * 255.0
    plt.imshow(image.astype('uint8'))
    plt.show()

# --- Prediction and Heatmap Functions ---
def predict(model, img, video_file_name=""):
    model.eval() # Set model to evaluation mode (disable dropout, batch norm updates)
    with torch.no_grad(): # Disable gradient calculations for inference
        fmap, logits = model(img.to(device))
    
    # Get the image for display from the last frame of the input sequence
    # For a single video (batch size 1), img is (1, seq_len, C, H, W)
    img_display_tensor = img[0, -1, :, :, :].unsqueeze(0) # Select the last frame of the video and add batch dim
    img_display_np = im_convert(img_display_tensor, video_file_name) # Convert to NumPy image

    logits = sm(logits) # Apply Softmax to get probabilities
    _, prediction_idx = torch.max(logits, 1) # Get the predicted class index
    confidence = logits[:, int(prediction_idx.item())].item() * 100
    
    print(f'Confidence of prediction: {confidence:.2f}%')
    return [int(prediction_idx.item()), confidence]

def plot_heat_map(frame_idx_in_seq, model, video_input_tensor, video_file_name=''):
    model.eval()
    with torch.no_grad():
        # video_input_tensor is (1, sequence_length, C, H, W)
        fmap, logits = model(video_input_tensor.to(device))

    # The `fmap` here will be of shape (sequence_length, channels, H_fmap, W_fmap)
    # We need to select the feature map corresponding to the specific frame_idx_in_seq
    current_fmap = fmap[frame_idx_in_seq].detach().cpu().numpy() # Shape: (channels, H_fmap, W_fmap)

    # Get the weights for the predicted class from the final linear layer
    weight_softmax = model.linear1.weight.detach().cpu().numpy() # Shape: (num_classes, hidden_dim)

    # Apply softmax to model logits to get the most confident prediction
    logits_softmax = sm(logits)
    _, prediction_idx = torch.max(logits_softmax, 1)
    predicted_class_idx = int(prediction_idx.item())

    # Ensure the dimensions match for the dot product
    # current_fmap has shape (channels, H_fmap, W_fmap)
    # weight_softmax[predicted_class_idx, :] has shape (hidden_dim,)
    # If channels and hidden_dim are the same (2048 as per your model), proceed.
    nc, h_fmap, w_fmap = current_fmap.shape # nc is channels, which is hidden_dim (2048)

    # Calculate CAM heatmap: sum of feature maps weighted by classifier weights
    # Reshape feature map from (channels, H_fmap, W_fmap) to (H_fmap * W_fmap, channels)
    # Then dot product with (channels,) weights results in (H_fmap * W_fmap,)
    cam = np.dot(current_fmap.reshape(nc, h_fmap * w_fmap).T, weight_softmax[predicted_class_idx, :].T)
    
    cam = cam.reshape(h_fmap, w_fmap)
    cam = np.maximum(cam, 0) # Apply ReLU to CAM (only positive contributions)
    cam = cam / (cam.max() + 1e-8) # Normalize to 0-1 range (add small epsilon to avoid division by zero)
    cam = np.uint8(255 * cam) # Scale to 0-255

    # Resize heatmap to original image dimensions for overlay
    heatmap_resized = cv2.resize(cam, (im_size, im_size))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Get the original frame for overlay
    original_frame_tensor = video_input_tensor[0, frame_idx_in_seq, :, :, :].unsqueeze(0)
    original_frame_np = im_convert(original_frame_tensor, f"{video_file_name}_original_frame_{frame_idx_in_seq}") # Convert to HWC, RGB, 0-255

    # Convert original frame to BGR for cv2.addWeighted if it's RGB
    original_frame_bgr = cv2.cvtColor(original_frame_np, cv2.COLOR_RGB2BGR)

    # Overlay heatmap on original image
    # result = cv2.addWeighted(src1, alpha, src2, beta, gamma)
    # alpha + beta should ideally be around 1.0 for balanced blend
    result_overlay = cv2.addWeighted(original_frame_bgr, 0.8, heatmap_colored, 0.5, 0) # Use BGR for overlay

    # Save the heatmap image
    heatmap_output_name = f"{video_file_name}_heatmap_frame_{frame_idx_in_seq}.png"
    image_save_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_images', heatmap_output_name)
    cv2.imwrite(image_save_path, result_overlay)

    return heatmap_output_name # Return filename for URL construction


# --- Model Selection ---
def get_accurate_model(sequence_length):
    model_name_candidates = []

    # --- DEBUGGING PRINTS START ---
    print(f"\n--- DEBUG: Entering get_accurate_model ---")
    print(f"DEBUG: Requested sequence_length: {sequence_length}")
    print(f"DEBUG: settings.BASE_DIR is: {settings.BASE_DIR}")
    models_dir_path = os.path.join(settings.BASE_DIR, "models")
    print(f"DEBUG: Looking for models in directory: {models_dir_path}")
    # --- DEBUGGING PRINTS END ---

    list_models = glob.glob(os.path.join(models_dir_path, "*.pt"))

    # --- DEBUGGING PRINTS START ---
    if not list_models:
        print(f"DEBUG: glob.glob found NO .pt files in {models_dir_path}. Please check directory and file extensions.")
    else:
        print(f"DEBUG: glob.glob found these .pt files: {list_models}")
    # --- DEBUGGING PRINTS END ---

    for model_path in list_models:
        model_filename = os.path.basename(model_path)

        # --- DEBUGGING PRINTS START ---
        print(f"\nDEBUG: Processing model filename: '{model_filename}'")
        # --- DEBUGGING PRINTS END ---

        try:
            parts = model_filename.split("_")

            # --- DEBUGGING PRINTS START ---
            print(f"DEBUG: Filename split into parts: {parts}")
            print(f"DEBUG: Length of parts: {len(parts)}")
            if len(parts) > 2: # Check to prevent IndexError before accessing parts[2]
                print(f"DEBUG: parts[1] (accuracy candidate): '{parts[1]}'")
                print(f"DEBUG: parts[2] (expected 'acc'): '{parts[2]}'")
            if len(parts) > 4: # Check to prevent IndexError before accessing parts[4]
                print(f"DEBUG: parts[3] (sequence candidate): '{parts[3]}'")
                print(f"DEBUG: parts[4] (expected 'data'): '{parts[4]}'")
            # --- DEBUGGING PRINTS END ---

            # Adjust the condition based on your exact integer accuracy format
            # If your filename is like "model_87_acc_60_frames_FF_data.pt"
            # parts will be ['model', '87', 'acc', '60', 'frames', 'FF', 'data.pt']
            # len(parts) == 5
            # parts[2] == 'acc'

            # Let's adjust for 'seq.pt' if that's what glob returns
            if len(parts) >= 5 and parts[2] == 'acc' and parts[len(parts)-1].startswith('data'):
                data = int(parts[3])
                acc = float(parts[1]) # float() can handle integer strings like '87' -> 87.0

                # --- DEBUGGING PRINTS START ---
                print(f"DEBUG: Successfully parsed: accuracy={acc}, sequence={data}")
                # --- DEBUGGING PRINTS END ---

                if data == sequence_length:
                    # --- DEBUGGING PRINTS START ---
                    print(f"DEBUG: Found matching model for sequence length {sequence_length}: {model_filename}")
                    # --- DEBUGGING PRINTS END ---
                    model_name_candidates.append((acc, model_path))
            else:
                # --- DEBUGGING PRINTS START ---
                print(f"DEBUG: Model filename '{model_filename}' does NOT match expected format (e.g., model_ACCURACY_acc_SEQ_seq.pt).")
                # --- DEBUGGING PRINTS END ---

        except (ValueError, IndexError) as e:
            # --- DEBUGGING PRINTS START ---
            print(f"DEBUG: Skipping model {model_filename} due to a parsing error (ValueError/IndexError): {e}")
            # --- DEBUGGING PRINTS END ---
            pass

    # --- DEBUGGING PRINTS START ---
    print(f"\nDEBUG: Final model candidates found: {model_name_candidates}")
    # --- DEBUGGING PRINTS END ---

    if model_name_candidates:
        model_name_candidates.sort(key=lambda x: x[0], reverse=True)
        final_model_path = model_name_candidates[0][1]
        print(f"Selected model: {os.path.basename(final_model_path)} (Path: {final_model_path})")
        return final_model_path
    else:
        print(f"No model found for sequence length {sequence_length} after checking all candidates. Please verify model filenames and their location: {models_dir_path}")
        return None # Indicate no model found


# --- File Upload Utilities ---
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# --- Django Views ---
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        # Clear session data on fresh GET request to index page
        request.session.pop('file_name', None)
        request.session.pop('preprocessed_images', None)
        request.session.pop('faces_cropped_images', None)
        request.session.pop('heatmap_images', None)
        request.session.pop('sequence_length', None) # Also clear sequence length

        return render(request, index_template_name, {"form": video_upload_form})
    else: # POST request
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            
            video_file_ext = video_file.name.split('.')[-1].lower() # Ensure lowercase
            video_content_type = video_file.content_type.split('/')[0]

            if video_content_type not in settings.CONTENT_TYPES:
                video_upload_form.add_error("upload_video_file", "Only video files are allowed based on content type.")
                return render(request, index_template_name, {"form": video_upload_form})

            if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                video_upload_form.add_error("upload_video_file", f"Maximum file size {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f} MB exceeded.")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0.")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if not allowed_video_file(video_file.name):
                video_upload_form.add_error("upload_video_file", "Only video files with allowed extensions are permitted (.mp4, .gif, etc.).")
                return render(request, index_template_name, {"form": video_upload_form})
            
            # --- Directory creation using BASE_DIR for media/ and models/ ---
            # These directories should be directly inside your BASE_DIR
            # e.g., D:\deepfake_detector_project\deepfake_detector\media
            uploaded_videos_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_videos')
            uploaded_images_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')

            os.makedirs(uploaded_videos_dir, exist_ok=True) # Now safely creates if not exists
            os.makedirs(uploaded_images_dir, exist_ok=True)
            # --- End Directory creation ---

            saved_video_file_name = f"uploaded_file_{int(time.time())}.{video_file_ext}"
            saved_video_path = os.path.join(uploaded_videos_dir, saved_video_file_name)
            
            # Save the uploaded video file chunk by chunk
            try:
                with open(saved_video_path, 'wb') as vFile:
                    for chunk in video_file.chunks():
                        vFile.write(chunk)
            except IOError as e:
                video_upload_form.add_error("upload_video_file", f"Failed to save video file: {e}. Check server permissions.")
                return render(request, index_template_name, {"form": video_upload_form})
            
            request.session['file_name'] = saved_video_path
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})


def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session:
            return redirect("ml_app:home")

        video_file_path = request.session['file_name']
        sequence_length = request.session.get('sequence_length', 60)

        video_file_name = os.path.basename(video_file_path)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        
        # Construct the URL for the original video file for template display
        # This assumes media/uploaded_videos is directly under your MEDIA_ROOT
        original_video_url = os.path.join(settings.MEDIA_URL, 'uploaded_videos', video_file_name)

        # Load validation dataset (uses the ValidationDataset class)
        path_to_videos = [video_file_path]
        # Make sure ValidationDataset is correctly instantiated (which it is)
        video_dataset = ValidationDataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        # Load model
        model_path = get_accurate_model(sequence_length)
        if not model_path:
            return render(request, predict_template_name, {
                "model_not_found": True,
                'sequence_length': sequence_length,
                'model_search_path': os.path.join(settings.BASE_DIR, "models")
            })

        # Initialize model based on device (CPU/GPU)
        model = Model(2) # Num classes is 2 for Real/Fake
        if device == "cuda":
            model.cuda()
        
        # Load model state dictionary. map_location='cpu' is good practice for compatibility.
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval() # Set model to evaluation mode
        except Exception as e:
            print(f"Error loading model state dict from {model_path}: {e}")
            return render(request, cuda_full_template_name, {"error_detail": f"Failed to load model weights: {e}"})

        start_time = time.time()
        print("<=== | Started Video Processing for Previews | ===>")
        
        preprocessed_images_urls = []
        faces_cropped_images_urls = []
        
        # Open video capture for extracting frames for preview display
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file for preview processing: {video_file_path}")
            return render(request, predict_template_name, {"error_message": "Could not open video file for preview. It might be corrupted or in an unsupported format."})

        preview_frames = []
        for _ in range(sequence_length): # Only read up to sequence_length for previews
            ret, frame = cap.read()
            if ret:
                preview_frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames extracted for preview: {len(preview_frames)}")
        
        padding = 40
        faces_found_count = 0

        # Process each preview frame for saving and face cropping
        for i, frame in enumerate(preview_frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image (original frame after BGR to RGB conversion)
            image_name_preprocessed = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path_preprocessed = os.path.join(settings.MEDIA_ROOT, 'uploaded_images', image_name_preprocessed)
            pImage.fromarray(rgb_frame).save(image_path_preprocessed)
            preprocessed_images_urls.append(os.path.join(settings.MEDIA_URL, 'uploaded_images', image_name_preprocessed))

            # Face detection and cropping for preview
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                top, right, bottom, left = face_locations[0]
                
                # Safer cropping with bounds checking
                h, w, _ = frame.shape
                top = max(0, top - padding)
                left = max(0, left - padding)
                bottom = min(h, bottom + padding)
                right = min(w, right + padding)
                
                frame_face = frame[top:bottom, left:right]
                
                if frame_face.size > 0: # Ensure cropped image is not empty
                    rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
                    image_name_cropped = f"{video_file_name_only}_cropped_faces_{i+1}.png"
                    image_path_cropped = os.path.join(settings.MEDIA_ROOT, 'uploaded_images', image_name_cropped)
                    pImage.fromarray(rgb_face).save(image_path_cropped)
                    faces_cropped_images_urls.append(os.path.join(settings.MEDIA_URL, 'uploaded_images', image_name_cropped))
                    faces_found_count += 1
                else:
                    print(f"Warning: Cropped face for preview frame {i+1} resulted in empty image. Skipping saving cropped face.")
            else:
                print(f"No face detected in preview frame {i+1} for cropping.")


        print("<=== | Video Processing and Face Cropping for Previews Done | ===>")
        print(f"--- Preview processing time: {time.time() - start_time:.2f} seconds ---")

        # If no faces were detected in any of the preview frames, inform the user
        if faces_found_count == 0:
            return render(request, predict_template_name, {"no_faces": True, 'original_video': original_video_url})

        # Perform prediction using the model on the prepared dataset
        try:
            heatmap_images_urls = []
            
            # video_dataset[0] loads the first (and only) video in the list
            print("<=== | Started Prediction | ===>")
            prediction_result = predict(model, video_dataset[0], video_file_name_only)
            confidence = round(prediction_result[1], 1)
            output = "REAL" if prediction_result[0] == 1 else "FAKE"
            print(f"Prediction: {prediction_result[0]} ({output}) | Confidence: {confidence:.2f}%")
            print("<=== | Prediction Done | ===>")
            print(f"--- Total prediction time: {time.time() - start_time:.2f} seconds ---")

            # Uncomment the following block if you want to generate and display heatmaps
            # for j in range(sequence_length):
            #     # Ensure plot_heat_map can handle the specific frame extraction from video_dataset[0]
            #     # if your ValidationDataset.getitem always returns the same frames.
            #     # Otherwise, you might need a separate DataLoader for heatmaps.
            #     heatmap_filename = plot_heat_map(j, model, video_dataset[0], video_file_name_only)
            #     if heatmap_filename:
            #         heatmap_images_urls.append(os.path.join(settings.MEDIA_URL, 'uploaded_images', heatmap_filename))

            context = {
                'preprocessed_images': preprocessed_images_urls,
                'faces_cropped_images': faces_cropped_images_urls,
                'heatmap_images': heatmap_images_urls, # Will be empty if heatmap generation is commented out
                'original_video': original_video_url,
                'output': output,
                'confidence': confidence,
            }

            return render(request, predict_template_name, context)

        except Exception as e:
            # Catch general exceptions during the ML pipeline and render a specific error page
            print(f"Exception occurred during prediction: {e}", exc_info=True) # Print full traceback to console
            return render(request, cuda_full_template_name, {"error_detail": f"An error occurred during prediction: {e}. Please try a different video or reduce sequence length."})

# --- Error Handlers and About Page ---
def about(request):
    return render(request, about_template_name)

def handler404(request, exception):
    return render(request, error_404_template_name, status=404)

def cuda_full(request):
    return render(request, cuda_full_template_name)