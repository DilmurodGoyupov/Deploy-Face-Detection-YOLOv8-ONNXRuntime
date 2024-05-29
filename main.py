import onnxruntime as ort
import numpy as np
import cv2
import time
import argparse # cmd-ning argumentlari bilan ishlash uchun kerak

# cmd cuhun argumentlarini kiritib olamiz
parser = argparse.ArgumentParser(description="ONNX modeli yordamida yuzni aniqlashni ishga tushirish")
parser.add_argument('--video-path', type=str, default='0', help="Web camera uchun 0, video uchun esa unga bo'lgan yo'lni ko'rsating")
parser.add_argument('--use-gpu', action='store_true', help="GPU bo'lsa unda ishlatamiz")
args = parser.parse_args()

# Model saqlangan papkaga yo'l
model_path = 'runs/detect/train2/weights/best.onnx'

# ONNX Runtime session paremetrlarini kiritamiz
session_options = ort.SessionOptions()
session_options.enable_mem_pattern = True
session_options.enable_cpu_mem_arena = True
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Provayderlar 
EP_list = ['CPUExecutionProvider']
if args.use_gpu:
    EP_list.insert(0, 'CUDAExecutionProvider')

# modelni yuklab olish
try:
    ort_session = ort.InferenceSession(model_path, sess_options=session_options, providers=EP_list)
    print(f"Model muvaffaqiyatli yuklandi {model_path}")
except Exception as e:
    print(f"Modelni yuklashda xatolik: {e}")
    exit()

# kiruvchi rasmimizni o'lcahmini to'girlab olamiz [1 x 3 x 640 x 640]
def preprocess(image):
    # rasmni kerakli o'lchamga olib kelamiz
    img = cv2.resize(image, (640, 640))
    img = img.transpose(2, 0, 1)  
    img = img[np.newaxis, :, :, :].astype(np.float32)  
    img /= 255.0  # normalizatsiya
    return img

def postprocess(outputs, img_shape):
    boxes, scores = [], []
    detections = outputs[0]
    print(f"Detections output shape: {detections.shape}")
    for detection in detections[0].T:  # 8400x5 qilib olamiz
        confidence = detection[4] # detection[4] bu bizning aniqlik darajamiz
        if confidence > 0.5: # aniqlik darajasi 0.5 dan baland bo'lsa bbox larni olamiz
            x_center, y_center, width, height = detection[:4]
            # Yuqori chap va pastki o'ng burchaklarni kordinatalarini image_shape-ga nisbatan hisoblaymiz
            print(f"Detection kordinatalar: x_center={x_center}, y_center={y_center}, width={width}, height={height}, confidence={confidence}")
            x1 = int((x_center - width / 2) * img_shape[1] / 640)
            y1 = int((y_center - height / 2) * img_shape[0] / 640)
            x2 = int((x_center + width / 2) * img_shape[1] / 640)
            y2 = int((y_center + height / 2) * img_shape[0] / 640)
            # kordinatalar rasm ichidaligiga ishonch hosil qilamiz
            x1 = max(0, min(x1, img_shape[1]))
            y1 = max(0, min(y1, img_shape[0]))
            x2 = max(0, min(x2, img_shape[1]))
            y2 = max(0, min(y2, img_shape[0]))

            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)

    print(f"Rasm o'lchamiga mos kodrinatalar: {boxes}")
    print(f"Aniqlik darajalari: {scores}")


    # Non-Maximum Suppression (NMS)
    index = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)
    print(f"NMS indices: {index}")
    if len(index) > 0:
        nms_boxes = [boxes[i] for i in index.flatten()]
        nms_scores = [scores[i] for i in index.flatten()]
    else:
        nms_boxes = []
        nms_scores = []

    return nms_boxes, nms_scores

# bboxlarni chizamiz
def draw_boxes(image, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# # Kamerani yoki videoni ulab olamiz
if args.video_path == '0':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    start_time = time.time()

    # rasmni o'lchamlarini to'g'irlab olamiz
    input_image = preprocess(frame)
    
    # Inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)

    # Chiqish ma'lumotlari bilan ishlash(bbox, scores)
    boxes, scores = postprocess(ort_outs, frame.shape)

    
    # rasmga to'g'ri-to'rtburchaklarni chizish
    draw_boxes(frame, boxes, scores)
    
    # frameni ko'rish
    cv2.imshow("Face Detection", frame)
    
    end_time = time.time()
    detection_time = end_time - start_time
    print(f"Face detection time: {detection_time:.4f} seconds")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()