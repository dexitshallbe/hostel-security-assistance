from ultralytics import YOLO


class YoloPersonGate:
    def __init__(self, model_path: str, conf: float = 0.35, person_class_id: int = 0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.person_class_id = person_class_id

    def person_count(self, frame_bgr) -> int:
        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        if not res:
            return 0
        boxes = res[0].boxes
        if boxes is None:
            return 0
        count = 0
        for c in boxes.cls.tolist():
            if int(c) == self.person_class_id:
                count += 1
        return count
