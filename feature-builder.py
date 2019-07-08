# Import packages
import cv2
import time
import numpy as np

# Global variables initializer
# Bounding box features relevant to fall detection
frame = 0 # Counter for average calculation
tc = 0 # Time counter (0 = start, 1 = stop)
avg_fps = []

W = [] # Array to calculate max and min width
H = [] # Array to calculate max and min height
P = [] # Array to calculate the max and min proportion (H / W)

Vw = [] # Width max velocity (n + 1 box - n)
Vh = [] # Height max velocity (n + 1 box - n)
Vp = [] # Proportion max velocity (n + 1 box - n)

# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture('video01.mp4')

# Read classes
classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Read pre-trained model and config file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Get output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, "%.4f" % (confidence), (x-10,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while True:
    # Read input
    ret,image = camera.read()
    frame += 1	    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # Create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop = False)
    # blob = cv2.resize(image, (416,416), cv2.INTER_LINEAR)

    # Set input blob for the network
    net.setInput(blob)

    # Run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # Initialization

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # For each detetion from each output layer get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Go through the detections remaining after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # print("Achei: ",classes[class_ids[i]], " Chance: %.4f" % confidences[i])
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        print('\nFrame: ', frame)
        print('Person: ', i)
        print('x: ', round(x))
        print('y: ', round(y))
        print('Width: ', round(w))
        print('Height: ', round(h))

        if (tc == 0):
        	start = time.time()
        	tc = 1
        else:
        	end = time.time()
        	avg_fps.append(1 / (end - start))
        	print('\nFPS: ', 1 / (end - start))
        	tc = 0

        if i == 0:
        	prop = round(h) / round(w)
        	W.append(round(w))
        	H.append(round(h))
        	P.append(prop)

    # Display output
    cv2.imshow("object detection", image)

    # Halt application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(0.5)

# Release resources
cv2.destroyAllWindows()

# Calculate and export features
max_W = np.amax(W)
min_W = np.amin(W)
avg_W = np.around(np.mean(W), decimals = 2)

max_H = np.amax(H)
min_H = np.amin(H)
avg_H = np.around(np.mean(H), decimals = 2)

max_P = np.around(np.amax(P), decimals = 4)
min_P = np.around(np.amin(P), decimals = 4)
avg_P = np.around(np.mean(P), decimals = 4)

# Time (in frames) for velocity calculations
# t = 1 means that velocity = difference between n frame and the n - 1 frame
t = int(np.around(1.5 * np.mean(avg_fps), decimals = 0))  # check velocity between 1.5 seconds (FPS SHOULD BE > 0.4)
i = 0

# Create velocity arrays
while (i + t) < len(W):
	Vw.append(abs(W[i + t] - W[i]))
	Vh.append(abs(H[i + t] - H[i]))
	Vp.append(np.around(abs(P[i + t] - P[i]), decimals = 4))
	i += 1

max_Vw = np.amax(Vw)
max_Vh = np.amax(Vh)
max_Vp = np.amax(Vp)

print('\n############################## FEATURES ##############################')

print('\nMax width: ', max_W)
print('Min width: ', min_W)
print('Avg width: ', avg_W)

print('\nMax height: ', max_H)
print('Min height: ', min_H)
print('Avg height: ', avg_H)

print('\nMax prop: ', max_P)
print('Min prop: ', min_P)
print('Avg prop: ', avg_P)

print('\nMax width velocity: ', max_Vw)
print('Max height velocity: ', max_Vh)
print('Max proportion velocity: ', max_Vp)