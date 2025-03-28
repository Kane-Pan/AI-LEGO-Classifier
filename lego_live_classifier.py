import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import os
import numpy as np

# Transform for resizing and normalizing images before feeding them to the model
input_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes, device):
    # Original model used resnet as a pretrained model to speed up training
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

def initialize_camera(device_index=0, width=1920, height=1080, fps=15):
    # Create the camera feed with given parameters
    cam = cv2.VideoCapture(device_index)

    if not cam.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, fps)
    return cam

def convert_frame_to_pil(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

'''Get the predicted LEGO ID and the AI's confidance of its answer using softmax'''
def get_prediction(pil_image, model, device, class_names):
    image_tensor = input_transforms(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_prob, pred_idx = torch.max(probabilities, 1)
        confidence = pred_prob.item() * 100
        predicted_ID = class_names[pred_idx.item()]
    return predicted_ID, confidence

def get_box_colour(confidence):
    ratio = max(0.0, min(confidence / 100.0, 1.0))

    # Red for low confidence. Green for higher confidence
    red = int(255 * (1 - ratio))
    green = int(255 * ratio)
    return (0, green, red)

'''area_threshold is the minimum pixel area required for a contour to be considered a valid LEGO
padding_ratio is the added ratio to the bounding square of the detected LEGO
returns cropped (the cropped image of the lego to be sent to the AI), largest (largest contour area),
and the bounding box parameters'''
def detect_lego(frame, area_threshold=500, padding_ratio=0.1):
    # Convert the frame to grayscale to emphasize darker colors
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get a new image by taking only the dark contours
    _, thresh_dark = cv2.threshold(gray_blur, 100, 255, cv2.THRESH_BINARY_INV)

    # Convert to HSV to make identifying brighter colors easier
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(hsv, (0, 50, 0), (179, 255, 255))

    # Combine the two masks, resulting in a binary image where dark regions are white and
    # bright regions are white, and the rest is black
    combined = cv2.bitwise_or(thresh_dark, mask_color)

    # Kernel set to 3x3 to fill in small gaps in regions (can change this if needed)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours in the binary image
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        valid = []
        height, width = frame.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Fixes a weird bug where contours are created at the edges of the frame
            if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
                valid.append(cnt)
        if valid:
            largest = max(valid, key=cv2.contourArea)

            # Create the bounding box (square) around the detected LEGO
            if cv2.contourArea(largest) > area_threshold:
                x, y, w, h = cv2.boundingRect(largest)
                pad_x = int(w * padding_ratio)
                pad_y = int(h * padding_ratio)
                x_new = max(x - pad_x, 0)
                y_new = max(y - pad_y, 0)
                w_new = min(w + 2 * pad_x, width - x_new)
                h_new = min(h + 2 * pad_y, height - y_new)
                cropped = frame[y_new:y_new+h_new, x_new:x_new+w_new]
                return cropped, largest, (x_new, y_new, w_new, h_new)
            
    return None, None, None

'''Helper function used to resize sample images for the UI'''
def resize_with_aspect_ratio(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))

'''The main loop that allows code to continuously update the UI based on what it is seeing through the 
camera stream. Takes in the AI model to be used for image classification, the device (cpu or cuda), the 
class names (LEGO IDs), and sample_images (a dictionary with keys being the LEGO ID and values the sample images)
Cam_index is by default 0 for system webcam for laptops but use 1 for external webcams'''
def live_LEGO_detection(model, device, class_names, sample_images, cam_index=0):
    cam = initialize_camera(device_index=cam_index)
    print("Press 'q' to exit the live feed.")
    predicted_ID, confidence = "None", 0.0
    # lego_colour = "Unknown"  # Initialize detected colour
    display_scale = 0.5
    panel_width = 300
    panel_height = 600  # Enough space for text and a small sample image

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Couldn't capture frame.")
            break

        cropped, contour, _ = detect_lego(frame)
        if cropped is not None:
            pil_cropped = convert_frame_to_pil(cropped)
            predicted_ID, confidence = get_prediction(pil_cropped, model, device, class_names)
            # Detect the color of the cropped LEGO piece
            # lego_colour = detect_colour(cropped)

        # Draw the outline of on the LEGO piece, coloured red to green from low to high confidence
        outline_colour = get_box_colour(confidence)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, outline_colour, 3)

        # Scale down the video feed for space for other things
        video_display = cv2.resize(frame, (int(frame.shape[1] * display_scale),
                                           int(frame.shape[0] * display_scale)))

        # Create a white panel for the text and sample image
        right_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255

        # Put text on the panel
        text1 = f"Predicted ID: {predicted_ID}"
        text2 = f"Confidence: {confidence:.2f}%"

        # text3 = f"Colour: {lego_color}"
        cv2.putText(right_panel, text1, (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(right_panel, text2, (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        # cv2.putText(right_panel, text3, (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)

        # Place a small sample image of the predicted LEGO below the text
        sample_img_path = sample_images.get(predicted_ID, None)
        if sample_img_path is not None:
            sample_img = cv2.imread(sample_img_path)
            if sample_img is not None:
                sample_img = resize_with_aspect_ratio(sample_img, panel_width - 20, panel_height - 160)
                sh, sw = sample_img.shape[:2]
                y_offset = 160
                x_offset = (panel_width - sw) // 2
                if y_offset + sh < panel_height:
                    right_panel[y_offset:y_offset+sh, x_offset:x_offset+sw] = sample_img

        max_height = max(video_display.shape[0], right_panel.shape[0])
        final_display = np.ones((max_height, video_display.shape[1] + right_panel.shape[1], 3), dtype=np.uint8) * 255

        # Place the video display
        final_display[:video_display.shape[0], :video_display.shape[1]] = video_display
        # Place the right panel
        final_display[:right_panel.shape[0], video_display.shape[1]:video_display.shape[1] + right_panel.shape[1]] = right_panel

        cv2.imshow("Live LEGO Detector", final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = r"" # Paste the file path to the model saved from the Trainer code
    renders_path = r"" # Paste the file path to one of the datasets used (they have the same class names)
    # Load dataset to get class names
    renders_dataset = datasets.ImageFolder(root=renders_path)
    class_names = renders_dataset.classes

    # Build dictionary for sample images (which will be the first image under each class)
    sample_images = {}
    for path, label in renders_dataset.imgs:
        cls_name = class_names[label]
        if cls_name not in sample_images:
            sample_images[cls_name] = path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=len(class_names), device=device)

    # Change cam_index to 0 if using a webcam that is a part of your computer. 1 for external USB webcam, for example
    live_LEGO_detection(model, device, class_names, sample_images, cam_index=1)
