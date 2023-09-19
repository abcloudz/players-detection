from ultralyticsplus import YOLO
import cv2
import colors
import threading


def get_color(i, cropped):
    thread_results[i] = colors.detect_color(cropped, field_color)[0]


def detect_colors(image, data, ratio):
    global thread_results
    objects = {}
    boxes = data.boxes.xyxy  # x1, y1, x2, y2
    scores = data.boxes.conf
    categories = data.boxes.cls

    threads = []
    for i, (score, label, box) in enumerate(zip(scores, categories, boxes)):
        if label.item() == 0.0:
            box = [int(i) for i in box.tolist()]
            x1, y1, x2, y2 = box
            cropped = image[y1:y2, x1:x2]
            ch, cw, _ = cropped.shape
            cropped = cv2.resize(cropped, (int(cw / ratio), int(ch / ratio)))
            cv2.imshow('Y1', cropped)
            # Create thread object with get_color target function
            threads.append(threading.Thread(target=get_color, args=(i, cropped)))
            objects[i] = {"object": "Person",
                          "score": round(score.item(), 3),
                          "location": box}

    thread_results = [(0, 0, 0)] * (max(objects.keys()) + 1)
    # Start all threads
    [t.start() for t in threads]
    print(f"{len(threads)} threads started", sep='\n')
    # Join all threads to the parent thread
    [t.join() for t in threads]
    print(f"{len(threads)} threads finished", sep='\n')

    # Distribute detected colors for objects
    for i in objects.keys():
        objects[i]["color"] = thread_results[i]

    all_colors = [color["color"] for color in objects.values()]

    # Get colour groups
    groups = colors.get_ranged_groups(all_colors, groups_color_filers)

    # Distribute objects by groups
    for key, value in objects.items():
        for k, v in groups.items():
            if value["color"] in v:
                objects[key]["team"] = k

    return image, objects


def draw_boxes(img, objects, ratio):
    for key, value in objects.items():
        try:
            x1, y1, x2, y2 = [int(i * ratio) for i in value['location']]
            # cv2.rectangle(image,(x1, y1),(x2, y2), (255, 255, 0), 5)   # highlight color detection mistakes
            cv2.rectangle(img, (x1, y1), (x2, y2), rect_color[value['team']], 5)
            cv2.putText(img, str(value['team']), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
            # Print objects parameters to the STDOUT
            print(
                f"Detected {value['object']} #{key:<3} "
                f"with score {value['score']:<5}, "
                f"team: {value['team']}, "
                f"color: {str(value['color']):<18}, "
                f"at location {value['location']} "
            )
        except KeyError as e:
            if str(e) == "'team'":
                print(f"Color {value['color']} doesnt match any team color range")
    return img


if __name__ == '__main__':
    # Define video source, file or stream
    cap = cv2.VideoCapture('sample.mp4')

    # Create team groups and adjust their color ranges. Each group can contain more than one color range
    # Required HSV parameters for each filter:
    # rgb_color, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value
    groups_color_filers = {"R": ((140, 250, 0, 1, 0, 1), (140, 280, 0, 0.45, 0, 0.2)),
                           "T1": ((251, 305, 0.2, 1, 0.3, 1),),
                           "T2": ((0, 360, 0, 0.15, 0.75, 1),),
                           "G": ((45, 64, 0.40, 1, 0.88, 1),)}

    # Adjust playground field color range according to the input colors
    field_color = (40, 150, 0.15, 1, 0.3, 0.8)

    # Define colors for obj rectangles
    rect_color = {'R': (0, 255, 255),
                  'T1': (255, 0, 0),
                  'T2': (0, 0, 255),
                  'G': (255, 0, 255)}

    # load model
    model = YOLO('ultralyticsplus/yolov8s')

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 100  # maximum number of detections per image

    # Process every 3rd frame. Used if high resolution causes slow motion
    counter, target = 0, 2

    # Processing video source in the loop
    while True:
        if counter == target:
            ret, frame = cap.read()
            counter = 0
            if not ret:
                print('Loop')
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Original frame resized for better visibility
            orig_frame = cv2.resize(frame, (1920, 1080))
            # Prepare low res frame for AI model recognition
            frame = cv2.resize(frame, (640, 360))

            # calculate aspect ratio for coordinates
            h0, w0, = orig_frame.shape[:2]
            h, w, = frame.shape[:2]
            aspect_ratio = 1
            if w0 / w == h0 / h:
                aspect_ratio = w0 / w

            # perform inference
            results = model.predict(frame)
            result = results[0]

            # processing objects and show result
            image, obj = detect_colors(frame, result, 1)
            out_image = draw_boxes(orig_frame, obj, aspect_ratio)
            cv2.imshow('YOLO', out_image)
        else:
            ret = cap.grab()
            counter += 1
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
