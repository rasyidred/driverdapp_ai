import cv2
import time
from ultralytics import YOLO

model =  YOLO("classification_model.pt")  

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    results = model.predict(frame, imgsz=640, device="cpu", verbose=False)
    annotated = results[0].plot()

    end = time.time()
    latency = (end - start) * 1000   # milliseconds
    fps = 1 / (end - start)

    cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(annotated, f"Latency: {latency:.1f} ms", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("YOLO Live Detection", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
