# Import our general libraries
import io
import math
import time
from typing import Tuple

import utils.tools as tool
from utils.cartesian import *

from codeproject_ai_sdk import LogVerbosity, ModuleRunner, JSON

from PIL import Image
import cv2
import numpy as np

from options import Options
from paddleocr import PaddleOCR

# Constants
debug_log       = False
no_plate_found  = 'Characters Not Found'

# Fallback for wide scenes: CodeProject.AI's plate detector is run at YOLO
# size=640, so very small plates in full 4K frames can disappear. If the
# normal full-frame pass misses, crop detected vehicles and try again.
vehicle_labels     = {"car", "truck", "bus", "motorcycle", "motorbike"}
vehicle_confidence = 0.25
max_vehicle_crops  = 3   # cap crops per frame; each miss fans out to this many x2 plate passes

# Globals
ocr                  = None
previous_label       = None
prev_avg_char_height = None
prev_avg_char_width  = None
resize_width_factor  = None
resize_height_factor = None
remove_spaces        = True
cropped_plate_dir    = None
save_cropped_plate   = False



def init_detect_platenumber(opts: Options) -> None:

    global ocr, resize_width_factor, resize_height_factor, remove_spaces, cropped_plate_dir, save_cropped_plate

    ocr = PaddleOCR(lang                = opts.language,
                    use_gpu             = opts.use_gpu,
                    show_log            = opts.log_verbosity == LogVerbosity.Loud,
                    det_db_unclip_ratio = opts.det_db_unclip_ratio,
                    det_db_box_thresh   = opts.box_detect_threshold,
                    drop_score          = opts.char_detect_threshold,
                    rec_algorithm       = opts.algorithm,
                    cls_model_dir       = opts.cls_model_dir,
                    det_model_dir       = opts.det_model_dir,
                    rec_model_dir       = opts.rec_model_dir,
                    use_angle_cls       = False)

    resize_width_factor  = opts.OCR_rescale_factor
    resize_height_factor = opts.OCR_rescale_factor
    remove_spaces        = opts.remove_spaces and not opts.OCR_training_dataset
    cropped_plate_dir    = opts.cropped_plate_dir
    save_cropped_plate   = opts.save_cropped_plate


async def call_image_api(module_runner: ModuleRunner, route: str, image: Image,
                         data: dict) -> Tuple[JSON, int]:

    with io.BytesIO() as image_buffer:
        image.save(image_buffer, format='JPEG') # 'PNG' - slow
        image_buffer.seek(0)
        image_data = ('image.jpeg', image_buffer, 'image/jpeg')

        start_time = time.perf_counter()
        response = await module_runner.call_api(route,
                                                files={ "image": image_data },
                                                data=data)
        inferenceMs = int((time.perf_counter() - start_time) * 1000)

    return response, inferenceMs


def clipped_rect(left: float, top: float, right: float, bottom: float,
                 image_width: int, image_height: int) -> Rect:

    left   = max(0, min(image_width,  int(left)))
    top    = max(0, min(image_height, int(top)))
    right  = max(0, min(image_width,  int(right)))
    bottom = max(0, min(image_height, int(bottom)))

    return Rect(left, top, right, bottom)


def rect_area(rect: Rect) -> int:
    return max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)


def detection_rect(detection: dict, image_width: int, image_height: int,
                   pad_x: float = 0.08, pad_y: float = 0.08) -> Rect:

    left   = float(detection["x_min"])
    top    = float(detection["y_min"])
    right  = float(detection["x_max"])
    bottom = float(detection["y_max"])
    width  = right - left
    height = bottom - top

    return clipped_rect(left   - width  * pad_x,
                        top    - height * pad_y,
                        right  + width  * pad_x,
                        bottom + height * pad_y,
                        image_width, image_height)


def vehicle_search_regions(vehicle: dict, image_width: int,
                           image_height: int) -> list:

    full_rect = detection_rect(vehicle, image_width, image_height)
    regions = [full_rect]

    vehicle_height = float(vehicle["y_max"]) - float(vehicle["y_min"])
    lower_top = float(vehicle["y_min"]) + vehicle_height * 0.35
    lower_rect = clipped_rect(float(vehicle["x_min"]) - vehicle_height * 0.08,
                              lower_top,
                              float(vehicle["x_max"]) + vehicle_height * 0.08,
                              float(vehicle["y_max"]) + vehicle_height * 0.08,
                              image_width, image_height)

    if rect_area(lower_rect) > 0:
        regions.append(lower_rect)

    return regions


def translate_plate_detection(plate_detection: dict, crop_rect: Rect,
                              image_width: int, image_height: int) -> dict:

    translated = dict(plate_detection)
    translated["x_min"] = max(0, min(image_width,  int(plate_detection["x_min"]) + crop_rect.left))
    translated["y_min"] = max(0, min(image_height, int(plate_detection["y_min"]) + crop_rect.top))
    translated["x_max"] = max(0, min(image_width,  int(plate_detection["x_max"]) + crop_rect.left))
    translated["y_max"] = max(0, min(image_height, int(plate_detection["y_max"]) + crop_rect.top))
    return translated


def rect_iou(first: dict, second: dict) -> float:

    left   = max(first["x_min"], second["x_min"])
    top    = max(first["y_min"], second["y_min"])
    right  = min(first["x_max"], second["x_max"])
    bottom = min(first["y_max"], second["y_max"])

    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0

    first_area  = max(0, first["x_max"] - first["x_min"]) * max(0, first["y_max"] - first["y_min"])
    second_area = max(0, second["x_max"] - second["x_min"]) * max(0, second["y_max"] - second["y_min"])
    union = first_area + second_area - intersection

    return intersection / union if union else 0


def dedupe_plate_predictions(predictions: list) -> list:

    predictions = sorted(predictions, key=lambda item: item.get("confidence", 0), reverse=True)
    deduped = []

    for prediction in predictions:
        if all(rect_iou(prediction, existing) < 0.35 for existing in deduped):
            deduped.append(prediction)

    return deduped


async def detect_plates_in_vehicle_crops(module_runner: ModuleRunner,
                                         opts: Options,
                                         pillow_image: Image) -> Tuple[list, int]:

    inferenceMs = 0
    image_width, image_height = pillow_image.size

    detect_vehicle_response, vehicleMs = await call_image_api(module_runner,
                                                              "vision/detection",
                                                              pillow_image,
                                                              {"min_confidence": vehicle_confidence})
    inferenceMs += vehicleMs

    if not "success" in detect_vehicle_response or not detect_vehicle_response["success"]:
        return [], inferenceMs

    vehicles = []
    for prediction in detect_vehicle_response.get("predictions", []):
        label = str(prediction.get("label", "")).lower()
        if label in vehicle_labels:
            vehicles.append(prediction)

    vehicles = sorted(vehicles, key=lambda item: item.get("confidence", 0), reverse=True)
    vehicles = vehicles[:max_vehicle_crops]

    plate_predictions = []

    for vehicle in vehicles:
        for crop_rect in vehicle_search_regions(vehicle, image_width, image_height):
            if rect_area(crop_rect) <= 0:
                continue

            crop = pillow_image.crop((crop_rect.left, crop_rect.top,
                                      crop_rect.right, crop_rect.bottom))
            detect_plate_response, plateMs = await call_image_api(module_runner,
                                                                  "vision/custom/license-plate",
                                                                  crop,
                                                                  {"min_confidence": opts.plate_confidence})
            inferenceMs += plateMs

            if not "success" in detect_plate_response or not detect_plate_response["success"]:
                continue

            crop_predictions = detect_plate_response.get("predictions", [])
            if crop_predictions:
                for plate_detection in crop_predictions:
                    plate_predictions.append(translate_plate_detection(plate_detection,
                                                                       crop_rect,
                                                                       image_width,
                                                                       image_height))
                break

    plate_predictions = dedupe_plate_predictions(plate_predictions)

    return plate_predictions, inferenceMs



async def detect_platenumber(module_runner: ModuleRunner, opts: Options, image: Image) -> JSON:

    """
    Performs the plate number detection
    Returns a tuple containing the JSON description of what was found, along 
    """

    global previous_label, prev_avg_char_width, prev_avg_char_height
    global resize_width_factor, resize_height_factor
    global cropped_plate_dir, save_cropped_plate
    
    outputs      = []
    pillow_image = image

    inferenceMs: int = 0

    # Look for plates in the full image first. If the plate model misses in a
    # wide scene, fall back to vehicle crops so tiny plates survive YOLO resize.
    try:
        detect_plate_response, plateMs = await call_image_api(module_runner,
                                                              "vision/custom/license-plate",
                                                              pillow_image,
                                                              {"min_confidence": opts.plate_confidence})
        inferenceMs += plateMs

        if not "success" in detect_plate_response or not detect_plate_response["success"]:
            message = detect_plate_response["error"] if "error" in detect_plate_response \
                                                     else "Unable to find plate"
            return { "error": message, "inferenceMs": inferenceMs }

        # Note: we will only get plates that have at least opts.plate_confidence
        # confidence.
        if not "predictions" in detect_plate_response or not detect_plate_response["predictions"]:
            fallback_predictions, fallbackMs = await detect_plates_in_vehicle_crops(module_runner,
                                                                                   opts,
                                                                                   pillow_image)
            inferenceMs += fallbackMs
            if not fallback_predictions:
                return { "predictions": [], "inferenceMs": inferenceMs }

            detect_plate_response["predictions"] = fallback_predictions

    except Exception as ex:
        await module_runner.report_error_async(ex, __file__)
        return {
            "error": f"Error trying to locate license plate ({ex.__class__.__name__})",
            "inferenceMs": inferenceMs
        }

    # We have a plate (or plates) detected, so let's prep the original image for some work
    numpy_image = np.array(pillow_image)

    # Remember: numpy is left handed when it comes to indexes
    orig_image_size: Size = Size(width = numpy_image.shape[1], height = numpy_image.shape[0])

    # Correct the colour space
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    # numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)

    # If a plate is found we'll pass this onto OCR
    for plate_detection in detect_plate_response["predictions"]:
        
        class_id = 0 if plate_detection["label"] == "DayPlate" else 1
                
        # Pull out just the detected plate.
        # The coordinates... (relative to the original image)
        plate_rect  = Rect(plate_detection["x_min"], plate_detection["y_min"],
                           plate_detection["x_max"], plate_detection["y_max"])
        # The image itself... (Its coordinates are now relative to itself)
        numpy_plate = numpy_image[plate_rect.top:plate_rect.bottom, plate_rect.left:plate_rect.right]

        # Store the size of the scaled plate before we start rotating it
        plate_size = Size(numpy_plate.shape[1], numpy_plate.shape[0])

        # Work out the angle we need to rotate the image to de-skew it (or use the manual override).
        if opts.auto_plate_rotate and not opts.OCR_training_dataset:
            angle       = 0
            angle_count = 0
            res         = None

            bounding_box_result = None
            try:
                bounding_box_result = ocr.ocr(numpy_plate, rec=False, cls=False)           
                for box in range(len(bounding_box_result)):
                    res = bounding_box_result[box]
                    if res:
                        if len(res) == 4 and len(res[0]) == 2:
                            x1, y1 = res[0][0], res[0][1]
                            x2, y2 = res[1][0], res[1][1]
                            angle += tool.calculate_angle(x1, y1, x2, y2)
                            angle_count += 1
                        else:
                            for line in res:
                                x1, y1 = line[0][0], line[0][1]
                                x2, y2 = line[1][0], line[1][1]
                                angle += tool.calculate_angle(x1, y1, x2, y2)
                                angle_count += 1

                                x1, y1 = line[3][0], line[3][1]
                                x2, y2 = line[2][0], line[2][1]
                                angle += tool.calculate_angle(x1, y1, x2, y2)
                                angle_count += 1

                plate_rotate_deg = angle / angle_count if angle_count > 0 else 0
            
                if debug_log:
                    with open("log.txt", "a") as text_file:
                        text_file.write(str(plate_rotate_deg) + "  " + str(angle_count) + "\n" + "\n")
            except:
                plate_rotate_deg = 0
            
        else:
            plate_rotate_deg = opts.plate_rotate_deg

        # If we need to rotate, then check to ensure that rotating the image won't chop off
        # important parts of the plate. Note that this will only happen to plates that are at,
        # or very close to, the edge of the image.
        if plate_rotate_deg and not opts.OCR_training_dataset:
        
            # We start with the assumption that we have a plate that is not displayed level. Once
            # the plate is de-skewed, the bounding box of the rotated plate will be *smaller* than
            # the skewed plate.
            
            # Calculate the corrected (smaller) bounding box if we were to rotate the image
            rotated_size: Size = tool.largest_rotated_rect(plate_size, math.radians(plate_rotate_deg))

            # Calculate the space between the corresponding edges of the (smaller) rotated plate
            # and the original plate.
            # buffer: Size = (plate_size - rotated_size) / 2.0
            buffer: Size = (plate_size - rotated_size).__div__(1.0)

            # 'buffer' represents the width/height of an area along the edges of the original image.
            #
            #         =============================
            #         |       buffer              |   
            #         |  ----------------------   |
            #         |  |                    |   |
            #         |  |    (safe area)     |   |
            #         |  |                    |   |
            #         |  ----------------------   |
            #         |       (unsafe area)       |
            #         =============================
            #
            # If a plate is detected in the original image, and part of that plate extends into 
            # (or over) the buffer area then it means that, on rotation, part of the image could 
            # be cut off. We should avoid that since it could cut off characters.

            # Calculate the region inside the extracted plate that the rotated plate image must fit
            # inside for us to go ahead and do the rotation (left, top, right, bottom)
            safe_rotation_area: Rect = Rect(buffer.width,  buffer.height,
                                            orig_image_size.width  - buffer.width,
                                            orig_image_size.height - buffer.height)

            # If the plate image isn't completely contained within the 'safe' area then do not
            # rotate
            if not safe_rotation_area.contains(plate_rect):
                plate_rotate_deg = 0

        # Rotate if we've determined we have a valid rotation angle to apply
        if plate_rotate_deg and not opts.OCR_training_dataset:
            numpy_plate = tool.rotate_image(numpy_plate, plate_rotate_deg, rotated_size)

        if opts.OCR_optimization and not opts.OCR_training_dataset:
            # Based on the previous observation we'll adjust the resize factor so that we get
            # closer and closer to an "optimal" factor that produces images whose characters
            # match our optimum character size. 
            # Assumptions:
            #  1. Most plates we detect will be more or less the same size. Obviously an issue if
            #     you are scanning plates both near and far
            #  2. The aspect ratio of the license plate text (width:height) is around 3:5
            if previous_label and previous_label != no_plate_found: 
                
                if prev_avg_char_width < opts.OCR_optimal_character_width and resize_width_factor < 50:
                    resize_width_factor += 0.02

                if prev_avg_char_width > opts.OCR_optimal_character_width and resize_width_factor > 0.03:
                    resize_width_factor -= 0.02
                    
                if prev_avg_char_height < opts.OCR_optimal_character_height and resize_height_factor < 50:
                    resize_height_factor += 0.02

                if prev_avg_char_height > opts.OCR_optimal_character_height and resize_height_factor > 0.03:
                    resize_height_factor -= 0.02

            numpy_plate = cv2.resize(numpy_plate, None, fx = resize_width_factor, 
                                     fy = resize_height_factor, interpolation = cv2.INTER_CUBIC)
        else:
            if opts.OCR_rescale_factor != 1.0:
                numpy_plate = tool.resize_image(numpy_plate, opts.OCR_rescale_factor * 100)

        if debug_log:
            with open("log.txt", "a") as text_file:
                text_file.write(f"{resize_height_factor}x{resize_width_factor} - {prev_avg_char_height}x{prev_avg_char_width}\n\n")

        """
        dimensions: Size = Size(numpy_plate.shape[1], numpy_plate.shape[0])
        dimensions.integerize()

        # Exaggerate the width to make line detection more prominent
        dimensions.width *= 1.5
        """
   
        # Pre-processing of the extracted plate to give the OCR a better chance of success
        
        # Run it through a super-resolution module to improve readability (TBD, but could be slow)
        # numpy_plate = super_resolution(numpy_plate)

        # resize image if required
        # if opts.OCR_rescale_factor != 1:
            # numpy_plate = tool.resize_image(numpy_plate, opts.OCR_rescale_factor * 100)

        # perform otsu thresh (best to use binary inverse since opencv contours work better with white text)
        # ret, numpy_plate = cv2.threshold(numpy_plate, 0, 255, cv2.THRESH_OTSU   | cv2.THRESH_BINARY_INV)
        # ret, numpy_plate = cv2.threshold(numpy_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # numpy_plate = cv2.GaussianBlur(numpy_plate, (5,5), 0)
        # numpy_plate = cv2.medianBlur(numpy_plate, 3)
        # numpy_plate = cv2.bilateralFilter(numpy_plate,9,75,75)
 
        # Read plate
        (label, confidence, avg_char_width, avg_char_height, plateInferenceMs, merged_bounding_boxes) = \
                            await read_plate_chars_PaddleOCR(module_runner, numpy_plate)
        inferenceMs += plateInferenceMs

        # Read the plate. This may require multiple attempts
        # If we had no success reading the original plate, apply some image enhancement
        # and try again
        """
        if label == no_plate_found:
            # If characters are not found try gamma correction and equalize
            # numpy_plate = cv2.fastNlMeansDenoisingColored(numpy_plate, None, 10, 10, 7, 21)
            numpy_plate = tool.gamma_correction(numpy_plate)
            # numpy_plate = tool.equalize(numpy_plate)
            # numpy_plate = cv2.cvtColor(numpy_plate, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("alpr-enhanced.jpg", numpy_plate)

            # Read plate, 2nd attempt
            (label, confidence, avg_char_width, avg_char_height, plateInferenceMs) = \
                             await read_plate_chars_PaddleOCR(module_runner, numpy_plate)
            inferenceMs += plateInferenceMs
        """

        if opts.save_cropped_plate:
            filename = f"{cropped_plate_dir}/alpr.jpg"
            cv2.imwrite(filename, numpy_plate)
            
        if merged_bounding_boxes is not None and "bounding boxes" not in merged_bounding_boxes:
            if opts.OCR_training_dataset:
                count = 0
                for box, (_, _) in merged_bounding_boxes:
                    count += 1
                    result = np.array(list(map(tuple, box)))
                    ocr_plate = tool.four_point_transform(numpy_plate, result)
                    filename = f"{cropped_plate_dir}/ocr{count}.jpg"
                    cv2.imwrite(filename, ocr_plate)


        if label and confidence:
            # Store to help with adjusting for next detection
            previous_label       = label
            prev_avg_char_width  = avg_char_width
            prev_avg_char_height = avg_char_height
            
            x_min, y_min = plate_detection["x_min"], plate_detection["y_min"]
            x_max, y_max = plate_detection["x_max"], plate_detection["y_max"]
            
            x_center = ((x_min + x_max) / 2) / orig_image_size.width
            y_center = ((y_min + y_max) / 2) / orig_image_size.height
            width = (x_max - x_min) / orig_image_size.width
            height = (y_max - y_min) / orig_image_size.height
            
            plate_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            # return what we found
            detection = {
                "confidence": confidence,
                "label": "Plate: " + label,
                "plate": label,
                "x_min": plate_rect.left,
                "y_min": plate_rect.top,
                "x_max": plate_rect.right,
                "y_max": plate_rect.bottom,
                "plate_annotation": plate_annotation,
                "ocr_annotation": merged_bounding_boxes,
                "valid_ocr_annotation": opts.OCR_training_dataset
            }
            outputs.append(detection)
        else:
            # Next loop around we don't want to needlessly adjust resize factors
            previous_label = no_plate_found

    return { "predictions": outputs, "inferenceMs": inferenceMs }


async def read_plate_chars_PaddleOCR(module_runner: ModuleRunner, image: Image) -> Tuple[str, float, int, int, float]:
    
    """
    This uses PaddleOCR for reading the plates. Note that the image being passed
    in should be a tightly cropped licence plate, so we're looking for the largest
    text box and will assume that's the plate number.
    Returns (plate label, confidence, avg char height (px), avg char width (px), inference time (ms))
    """

    global remove_spaces
    inferenceTimeMs: int = 0

    try:
        start_time = time.perf_counter()
        ocr_response = ocr.ocr(image, cls=False)
        inferenceTimeMs = int((time.perf_counter() - start_time) * 1000)

        # Note that ocr_response[0][0][0][0] could be a float with value 0 ('false'), or in some
        # other universe maybe it's a string. To be really careful we would have a test like
        # if hasattr(ocr_response[0][0][0][0], '__len__') and (not isinstance(ocr_response[0][0][0][0], str))
        if not ocr_response or not ocr_response[0] or not ocr_response[0][0] or not ocr_response[0][0][0]:
            return no_plate_found, 0, 0, 0, inferenceTimeMs, None

        # Seems that different versions of paddle return different structures, OR
        # paddle returns different structures depending on its mood. We're expecting
        # ocr_response = array of single set of detections
        #  -> detections = array of detection
        #    -> detection = array of bounding box, classification
        #       -> bounding box = array of [x,y], classification = array of label, confidence.
        # so ocr_response[0][0][0][0][0] = first 'x' of the bounding boxes, which is a float
        # However, the first "array of single set of detections" isn't always there, so first
        # check to see if ocr_response[0][0][0][0] is a float

        detections = ocr_response if isinstance(ocr_response[0][0][0][0], float) else ocr_response[0]

        if debug_log:
            with open("log.txt", "a") as text_file:
                text_file.write(str(ocr_response) + "\n" + "\n")

        # Find the biggest textbox and assume that's the plate number
        plate_label, plate_confidence, avg_char_width, avg_char_height, merged_bounding_boxes = tool.merge_text_detections(detections, remove_spaces)

        if not plate_label:
            return no_plate_found, 0, 0, 0, inferenceTimeMs, None

        return plate_label, plate_confidence, avg_char_height, avg_char_width, inferenceTimeMs, merged_bounding_boxes

    except Exception as ex:
        module_runner.report_error_async(ex, __file__)    
        return None, 0, 0, 0, inferenceTimeMs, None
