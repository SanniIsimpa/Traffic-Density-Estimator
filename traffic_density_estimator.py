import sys
import time
from collections import defaultdict
import cv2
from ultralytics import YOLO

MODEL_WEIGHTS = "yolov8n.pt"
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
CONF_THRESHOLD = 0.35
INFERENCE_WIDTH = 640
TRACKER_CFG = "bytetrack.yaml"
DENSITY_THRESHOLDS = {"Low": 5, "Moderate": 12}

def load_model(weights_path: str = MODEL_WEIGHTS) -> YOLO:
    model = YOLO(weights_path)
    return model

def resolve_target_class_ids(model: YOLO, target_names: set[str]) -> set[int]:
    name_to_id = {name: idx for idx, name in model.names.items()}
    target_ids = {name_to_id[name] for name in target_names if name in name_to_id}
    return target_ids

def process_frame(model: YOLO, frame, allowed_ids: set[int]):
    results = model.track(
        source=frame,
        persist=True,
        tracker=TRACKER_CFG,
        classes=list(allowed_ids),
        conf=CONF_THRESHOLD,
        imgsz=INFERENCE_WIDTH,
        verbose=False,
    )
    return results[0]

def draw_annotations(frame, result, model: YOLO, unique_ids_seen: set[int]):
    per_class_now = defaultdict(int)
    boxes = result.boxes
    has_tracks = boxes is not None and boxes.id is not None

    if has_tracks:
        xyxys = boxes.xyxy.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), track_id, cls_id, conf in zip(xyxys, track_ids, class_ids, confs):
            class_name = model.names[int(cls_id)]
            per_class_now[class_name] += 1
            unique_ids_seen.add(int(track_id))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"ID {track_id} {class_name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (0, 200, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    total_now = sum(per_class_now.values())
    if total_now <= DENSITY_THRESHOLDS["Low"]:
        density_label, density_color = "LOW", (0, 200, 0)
    elif total_now <= DENSITY_THRESHOLDS["Moderate"]:
        density_label, density_color = "MODERATE", (0, 200, 255)
    else:
        density_label, density_color = "HEAVY", (0, 0, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"Density: {density_label} ({total_now})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "  ".join(f"{k}:{v}" for k, v in sorted(per_class_now.items())), (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Unique total: {len(unique_ids_seen)}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, per_class_now, density_label

def main_loop(source) -> None:
    model = load_model()
    allowed_ids = resolve_target_class_ids(model, VEHICLE_CLASSES)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"Could not open: {source!r}")
    unique_ids_seen = set()
    prev_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        result = process_frame(model, frame, allowed_ids)
        frame, _, _ = draw_annotations(frame, result, model, unique_ids_seen)
        
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:4.1f}", (frame.shape[1] - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Traffic Density Estimator", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    source = "traffic_video.mp4" 
    main_loop(source)