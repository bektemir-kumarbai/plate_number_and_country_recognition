import torch
from init_models import country_model, upsampler
import cv2
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
import time

_paddle_ocr_cache = {}

SYMBOLS = {
    "-", "#", "_", "+", "=", "!", "@", "$", "%", "*", "&", "(", ")", "^", "/", "|", ";", ":", ".", ",", "·", "<", ">", "[", "]"
}
PLATE_TYPES = {
    0: "numberplate", 1: "brand_numberplate",
    2: "filled_numberplate", 3: "empty_numberplate"
}

start_time = time.time()
number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                           path_to_model="modelhub://yolov11x_brand_np",
                                           image_loader="opencv")
print(f"Pipeline initialization took {time.time() - start_time:.2f} seconds")

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

@measure_time
def paddle(img, lang):

    if lang in _paddle_ocr_cache:
        recognition = _paddle_ocr_cache[lang]
    else:
        recognition = PaddleOCR(use_angle_cls=True, lang=lang, det=False)
        _paddle_ocr_cache[lang] = recognition
    
    try:
        result = recognition.ocr(img)
        
        if not result or not result[0]:
            return "", []
            
        results_array = []
        confidences_array = []
        for idx in range(len(result[0])):
            res = result[0][idx][1][0]
            confidence = result[0][idx][1][1]
            results_array.append(res)
            confidences_array.append(confidence)
            
        combined_element = "".join(results_array)
        final = combined_element.replace(" ", "")
        return final, confidences_array
    except Exception as e:
        print(f"Error in paddle OCR: {e}")
        return "", []

# Define a function for preprocessing an image
@measure_time
def preprocess(img):
    # Resize image if too large to improve processing speed
    h, w = img.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    
    # Detect rotation angle
    rot_angle = 0
    
    # Use cv2 for grayscale conversion (faster than rgb2gray)
    if len(img.shape) == 3:
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    else:
        grayscale = img / 255.0
    
    # Apply edge detection
    edges = canny(grayscale, sigma=3.0)
    
    # Calculate rotation using Hough transform
    out, angles, distances = hough_line(edges)
    _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
    
    if len(angles_peaks) > 0:
        angle = np.mean(np.rad2deg(angles_peaks))
        
        # Adjust rotation angle based on orientation
        if 0 <= angle <= 90:
            rot_angle = angle - 90
        elif -45 <= angle < 0:
            rot_angle = angle - 90
        elif -90 <= angle < -45:
            rot_angle = 90 + angle
            
        # Skip rotation if angle is too extreme
        if abs(rot_angle) > 20:
            rot_angle = 0
    
    # Only rotate if angle is significant
    if abs(rot_angle) > 1.0:
        rotated = rotate(img, rot_angle, resize=True) * 255
        rotated = rotated.astype(np.uint8)

        H, W = rotated.shape[:2]
        crop_margin = 0
        if W / H > 1.5:
            crop_margin = np.abs(int(np.sin(np.radians(rot_angle)) * H))
        elif W / H < 0.8:  
            crop_margin = np.abs(int(np.sin(np.radians(rot_angle)) * W))

        # Apply the Crop if margin is significant
        if crop_margin > 6:
            crop_margin = min(crop_margin, min(H, W) // 4)  # Limit crop to 1/4 of image
            rotated = rotated[crop_margin:-crop_margin, crop_margin:-crop_margin]
            
        return rotated
    else:
        return img

@measure_time
def del_symbols(input_string):
    if not input_string:
        return ""
    return ''.join(char for char in input_string if char not in SYMBOLS)

@measure_time
def calculate_mean(numbers):
    if not numbers or len(numbers) == 0:
        return None
    return float(sum(numbers) / len(numbers))

@measure_time
def lp_det_reco(img_path):
    try:
        start_process = time.time()
        result = number_plate_detection_and_reading([img_path])

        print(f"Pipeline processing took {time.time() - start_process:.2f} seconds")

        (images, images_bboxs, images_points, images_zones, 
         region_ids, region_names, count_lines, confidences, texts) = unzip(result)

        if not images_bboxs or len(images_bboxs[0]) == 0:
            raise Exception("Plate number is not recognized")

        x_min, y_min, x_max, y_max, _, plate_type, _ = images_bboxs[0][0]
        if plate_type not in PLATE_TYPES or PLATE_TYPES[plate_type] != "numberplate":
            raise Exception("Plate number is not recognized")

        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        if count_lines[0][0] == 1 and len(images_zones[0]) > 0:
            pro = images_zones[0][0].astype("uint8")
        else:
            pro = preprocess(images[0][y_min:y_max, x_min:x_max])
        pro_resized = cv2.resize(pro, (224, 224))
        pred_country, pred_idx, probs_country = country_model.predict(pro_resized)
        country = (pred_country, probs_country[pred_idx])
        H, W, _ = pro.shape
        if 100 <= H <= 300 and 100 <= W <= 300:
            try:
                img_enh, _ = upsampler.enhance(pro, outscale=1)
            except Exception as e:
                print(f"Enhancement error: {e}")
                img_enh = pro
        else:
            img_enh = pro
        if country[0] == "CN":
            combined_element_without_spaces, conf = paddle(img_enh, "ch")
            combined_element_without_spaces = del_symbols(combined_element_without_spaces)
            if (combined_element_without_spaces and 
                (combined_element_without_spaces[0] == "0" or 
                 (len(combined_element_without_spaces) > 1 and combined_element_without_spaces[1] == "0"))):
                combined_element_without_spaces = combined_element_without_spaces.replace("0", "Q", 1)
            elif texts and texts[0]:
                combined_element_without_spaces = texts[0][0]
                
        elif country[0] == "KG":
            number_text = texts[0][0] if texts and texts[0] else ""
            
            if number_text:
                number_text = list(number_text)  # Convert to list for editing
                
                if number_text[0] == "G":
                    number_text[0] = "0"

                second_char_int = None
                if len(number_text) > 1 and number_text[1].isdigit():
                    second_char_int = int(number_text[1])
                
                if (number_text[0] == "0" and 
                    second_char_int is not None and 
                    second_char_int != 0 and 
                    len(number_text) >= 6):
                    number_text.insert(2, "KG")
            
            conf = confidences[0][0] if confidences and confidences[0] else 0.0
            combined_element_without_spaces = "".join(number_text) if number_text else ""
            
        elif country[0] in ["RU", "KZ"]:
            if count_lines[0][0] >= 2:
                combined_element_without_spaces, conf = paddle(img_enh, "en")
                combined_element_without_spaces = combined_element_without_spaces.replace("KZ", "")
            else:
                conf = confidences[0][0] if confidences and confidences[0] else 0.0
                combined_element_without_spaces = texts[0][0] if texts and texts[0] else ""
                
        elif country[0] == "UZ":
            combined_element_without_spaces, conf = paddle(img_enh, "en")
            
        else:
            conf = confidences[0][0] if confidences and confidences[0] else 0.0
            combined_element_without_spaces = texts[0][0] if texts and texts[0] else ""

        combined_element_without_spaces = del_symbols(combined_element_without_spaces)
        if isinstance(conf, list):
            conf = calculate_mean(conf)
            
    except Exception as error:
        print('error = ', error)
        combined_element_without_spaces = None
        conf = None
        country = (None, None)

    return {
        "license_plate_number": combined_element_without_spaces,
        "license_plate_number_score": conf,
        "license_plate_country": country[0] if isinstance(country, tuple) else None,
        "license_plate_country_score": country[1] if isinstance(country, tuple) else None,
    }
