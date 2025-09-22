import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import os
import numpy as np
import argparse

import time
from pyfirmata import Arduino, SERVO

from type_specification import type_hierarchy, get_category_from_id

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

def get_prediction_colour(pil_image, model, device, class_names):
    image_tensor = input_transforms(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_prob, pred_idx = torch.max(probabilities, 1)
        confidence = pred_prob.item() * 100
        predicted_colour = class_names[pred_idx.item()]
    return predicted_colour, confidence

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

    # Kernel set to 3x3 to fill in small gaps in regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
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

# LEGO distributor
def build_colour_to_angles(colour_string):
   
    colours = [col.strip().lower().capitalize() for col in colour_string.split(',')]
    num_colours = len(colours)
    
    # 8 available slots for the colours.
    base_slots = 8 // num_colours # Every colour gets at least this many slots.
    extra_slots = 8 % num_colours # Extra slots to distribute to.
    if num_colours > 8:
        print("Warning: Only 8 colours can be used. The last one(s) will be ignored.")
    
    allocated_slots = []
    for i, colour in enumerate(colours):
        count = base_slots + (1 if i < extra_slots else 0)
        allocated_slots.extend([colour] * count)
    
    # Build dictionary mapping each colour to a list of angles.
    colour_to_angles = {}
    for slot, colour in enumerate(allocated_slots):
        angle = slot * 40
        colour_to_angles.setdefault(colour, []).append(angle)
    return colour_to_angles

class ColourSlotSelector:
    """
    A helper class that uses round-robin selection of slot angles for each colour.
    """
    def __init__(self, colour_to_angles):
        self.colour_to_angles = colour_to_angles
        # Initialize a counter for each colour
        self.counters = {colour: 0 for colour in colour_to_angles}
    
    def get_next_angle(self, colour):
        
        if colour not in self.colour_to_angles:
            # For colours not specified, use "Other" category
            return 320
        angles = self.colour_to_angles[colour]
        index = self.counters[colour]
        angle = angles[index]

        # Update counter
        self.counters[colour] = (index + 1) % len(angles)
        return angle

class TypeSlotSelector:
    """
    A helper class that uses round-robin selection of slot angles for LEGO types.
    Allocates 3 slots for 'brick', 3 slots for 'tile', 2 slots for 'plate', 1 slot for 'other'
    """
    def __init__(self):
        self.type_to_angles = {
            'brick': [0, 40, 80],
            'tile': [120, 160, 200],
            'plate': [240, 280],
        }
        self.other_angle = 320
        self.counters = {lego_type: 0 for lego_type in self.type_to_angles}

    def get_next_angle(self, lego_type):
        if lego_type not in self.type_to_angles:
            return self.other_angle

        angles = self.type_to_angles[lego_type]
        index = self.counters[lego_type]
        angle = angles[index]

        # Update for next round
        self.counters[lego_type] = (index + 1) % len(angles)
        return angle

# Terminal arguments
parser = argparse.ArgumentParser(description="Colour customization for Lego sorter")
default_colours = "red,blue,green,yellow,brown,black,white,grey"
parser.add_argument('--customize_colours', type=build_colour_to_angles,
                    default=build_colour_to_angles(default_colours),
                    help="Provide a comma-separated list of colours, e.g. 'red,blue,green'")
parser.add_argument('--colour_model_path', type=str,
                    help="Go to the colour model in your file directory and right click it. Then, copy its path and paste it here")
parser.add_argument('--type_model_path', type=str,
                    help="Go to the type model in your file directory and right click it. Then, copy its path and paste it here")
parser.add_argument('--colour_data_path', type=str,
                    help="Go to the type data folder in your file directory and right click it. Then, copy its path and paste it here")
parser.add_argument('--type_data_path', type=str,
                    help="Go to the colour data folder in your file directory and right click it. Then, copy its path and paste it here")
args = parser.parse_args()

# Arduino
# Initialize Arduino (adjust COM port if needed)
board = Arduino('COM10')
time.sleep(2) 

servo1 = board.get_pin('d:9:s')
servo2 = board.get_pin('d:8:s')
servo_feeder1 = board.get_pin('d:10:s')
servo_feeder2 = board.get_pin('d:11:s')
servo_dropper = board.get_pin('d:12:s')

def vibrate_feeder():
    servo_feeder1.write(60)
    time.sleep(0.1)
    servo_feeder2.write(60)
    time.sleep(0.1)
    servo_feeder1.write(0)
    time.sleep(0.1)
    servo_feeder2.write(0)
    time.sleep(0.1)
    
def drop_lego():
    servo_dropper.write(120)
    time.sleep(0.6)
    servo_dropper.write(0)
    time.sleep(0.6)

def set_angle(angle):
    
    if angle <= 160:
        servo1.write(angle)
        time.sleep(0.067)
        servo2.write(0)
        time.sleep(0.067)
    else:
        servo1.write(180)
        time.sleep(0.067)
        servo2.write(angle - 180)
        time.sleep(0.067)

# Main Loop

def live_inference_loop(model_type, model_colour, device, class_names, colour_class_names, sample_images, cam_index=0, colour_mode=True, type_mode=False):

    if not colour_mode and not type_mode:
        raise ValueError("Please choose either colour or type mode.")
    if colour_mode and type_mode:
        raise ValueError("Colour and type mode cannot be applied at the same time.")

    cam = initialize_camera(device_index=cam_index)
    print("Press 'q' to exit the live feed.")
    predicted_ID, type_confidence = "None", 0.0
    predicted_colour, colour_confidence = "None", 0.0
    display_scale = 0.5
    panel_width = 300
    panel_height = 600  # space for text and a small sample image

    colour_to_angles = args.customize_colours
    colour_selector = ColourSlotSelector(colour_to_angles)
    type_selector = TypeSlotSelector()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Couldn't capture frame.")
            break

        cropped, contour, _ = detect_lego(frame)
        if cropped is not None:
            pil_cropped = convert_frame_to_pil(cropped)
            # Get the predicted ID from the type model.
            predicted_ID, type_confidence = get_prediction(pil_cropped, model_type, device, class_names)
            # Get the predicted colour from the colour model.
            predicted_colour, colour_confidence = get_prediction_colour(pil_cropped, model_colour, device, colour_class_names)

        # Draw the outline of the detected LEGO piece, colored from red to green based on confidence.
        outline_colour = get_box_colour(type_confidence)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, outline_colour, 3)

        # Scale down the video feed for display.
        video_display = cv2.resize(frame, (int(frame.shape[1] * display_scale),
                                           int(frame.shape[0] * display_scale)))

        # Create a white panel for text and a sample image.
        right_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255

        # Put text on the panel.
        text1 = f"Predicted ID: {predicted_ID}"
        text2 = f"Type Conf: {type_confidence:.2f}%"
        text3 = f"Colour: {predicted_colour}"
        text4 = f"Colour Conf: {colour_confidence:.2f}%"
        cv2.putText(right_panel, text1, (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(right_panel, text2, (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(right_panel, text3, (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(right_panel, text4, (10, 160), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)

        # Place a small sample image of the predicted LEGO piece (based on the type prediction).
        sample_img_path = sample_images.get(predicted_ID, None)
        if sample_img_path is not None:
            sample_img = cv2.imread(sample_img_path)
            if sample_img is not None:
                sample_img = resize_with_aspect_ratio(sample_img, panel_width - 20, panel_height - 200)
                sh, sw = sample_img.shape[:2]
                y_offset = 200
                x_offset = (panel_width - sw) // 2
                if y_offset + sh < panel_height:
                    right_panel[y_offset:y_offset+sh, x_offset:x_offset+sw] = sample_img

        # Combine the video display and the right panel into the final display.
        max_height = max(video_display.shape[0], right_panel.shape[0])
        final_display = np.ones((max_height, video_display.shape[1] + right_panel.shape[1], 3), dtype=np.uint8) * 255
        final_display[:video_display.shape[0], :video_display.shape[1]] = video_display
        final_display[:right_panel.shape[0], video_display.shape[1]:video_display.shape[1] + right_panel.shape[1]] = right_panel

        cv2.imshow("Live LEGO Detector", final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

         # Motors respond with required angle
        if predicted_colour != None and colour_confidence > 80 and colour_mode:
            target_angle = colour_selector.get_next_angle(predicted_colour)
            set_angle(target_angle)
            time.sleep(1.2)
            drop_lego()
            time.sleep(0.2)
            predicted_colour = None


        if predicted_ID != None and type_confidence > 80 and type_mode:
            target_angle = type_selector.get_next_angle(get_category_from_id(predicted_ID))
            set_angle(target_angle)
            time.sleep(1.2)
            drop_lego()
            time.sleep(0.2)
            predicted_ID = None

        # vibrate the feeder once after every run. stops after some lego is detected with high confidence
        vibrate_feeder()


    cam.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':

    # Paste the file path to the model
    model_type_path = r""
    model_colour_path = r""
    # Paste the file path to one of the datasets used (they have the same class names)
    types_path = r"" 
    colours_path = r""
    
    if model_type_path == "":
        model_type_path = args.type_model_path
    if model_colour_path == "":
        model_colour_path = args.colour_model_path
    if types_path == "":
        model_colour_path = args.type_data_path
    if colours_path == "":
        model_colour_path = args.colour_data_path
    

    # Load dataset to get class names
    renders_dataset = datasets.ImageFolder(root=types_path)
    colours_dataset = datasets.ImageFolder(root=colours_path)
    class_type_names = renders_dataset.classes
    class_colour_names = colours_dataset.classes

    # Build dictionary for sample images (which will be the first image under each class)
    sample_images = {}
    for path, label in renders_dataset.imgs:
        cls_name = class_type_names[label]
        if cls_name not in sample_images:
            sample_images[cls_name] = path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = load_model(model_type_path, num_classes=len(class_type_names), device=device)
    model_colour = load_model(model_colour_path, num_classes=len(class_colour_names), device=device)

    # Change cam_index to 0 if using a webcam that is a part of your computer. 1 for external USB webcam, for example
    live_inference_loop(model_type, model_colour, device, class_type_names, class_colour_names, sample_images, cam_index=1, colour_mode=True)
