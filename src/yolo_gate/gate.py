from typing import List, Tuple
from ultralytics import YOLO


class YoloPersonGate:
    def __init__(self, model_path: str, conf: float = 0.35, person_class_id: int = 0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.person_class_id = person_class_id

    def detect_person_boxes(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of person bboxes (x1,y1,x2,y2) for current frame.
        """
        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        if not res:
            return []
        boxes = res[0].boxes
        if boxes is None or boxes.xyxy is None:
            return []

        out = []
        cls_list = boxes.cls.tolist()
        xyxy = boxes.xyxy.tolist()
        for c, b in zip(cls_list, xyxy):
            if int(c) == self.person_class_id:
                x1, y1, x2, y2 = map(int, b)
                out.append((x1, y1, x2, y2))
        return out

    def person_count(self, frame_bgr) -> int:
        return len(self.detect_person_boxes(frame_bgr))