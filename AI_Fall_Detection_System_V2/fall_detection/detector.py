"""YOLO pose inference, multi-person tracking, and annotated rendering."""
from dataclasses import dataclass
import math
from typing import List, Tuple
import cv2
from ultralytics import YOLO
from .logic import TemporalFallTracker

@dataclass
class PersonDetection:
    person_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: object
    confidence: float
    is_fallen: bool
    just_fell: bool
    posture_score: float
    torso_angle: float
    aspect_ratio: float

class FallDetectionSystem:
    SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
                (11,13),(13,15),(12,14),(14,16)]
    def __init__(self, model_path="yolo11n-pose.pt", confidence=.35, image_size=640,
                 confirmation_seconds=.8, angle_threshold=55.,
                 horizontal_ratio_threshold=1.15, device=None):
        self.model = YOLO(model_path)
        self.confidence, self.image_size, self.device = confidence, image_size, device
        self.tracker = TemporalFallTracker(confirmation_seconds, angle_threshold=angle_threshold,
                                           horizontal_ratio_threshold=horizontal_ratio_threshold)

    @staticmethod
    def _angle(shoulder, hip):
        dx, dy = float(shoulder[0]-hip[0]), float(shoulder[1]-hip[1])
        angle = abs(math.degrees(math.atan2(dx, -dy)))
        return min(angle, 180-angle)

    def process(self, frame) -> List[PersonDetection]:
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                   conf=self.confidence, imgsz=self.image_size,
                                   verbose=False, device=self.device)
        if not results or results[0].boxes is None or results[0].keypoints is None:
            self.tracker.remove_stale()
            return []
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else None
        height, width = frame.shape[:2]
        detections = []
        for i, (box, kp) in enumerate(zip(boxes, points)):
            pid = int(ids[i]) if ids is not None and i < len(ids) else i+1
            x1,y1,x2,y2 = [int(v) for v in box]
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(width-1,x2),min(height-1,y2)
            bw,bh=max(1,x2-x1),max(1,y2-y1)
            ratio=bw/bh
            required=(5,6,11,12)
            if len(kp) <= 16 or any(kp[j,2] < .25 for j in required):
                angle=0.
            else:
                shoulder=(kp[5,:2]+kp[6,:2])/2
                hip=(kp[11,:2]+kp[12,:2])/2
                angle=self._angle(shoulder,hip)
            fallen,new_event,score,sangle,sratio=self.tracker.update(pid,angle,ratio)
            detections.append(PersonDetection(pid,(x1,y1,bw,bh),kp,float(confs[i]),
                                              fallen,new_event,score,sangle,sratio))
        self.tracker.remove_stale()
        return detections

def draw_detections(frame, detections):
    height,width=frame.shape[:2]
    for person in detections:
        x,y,w,h=person.bbox
        color=(0,0,255) if person.is_fallen else (45,220,45)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        pts=person.keypoints
        for a,b in FallDetectionSystem.SKELETON:
            if a<len(pts) and b<len(pts) and pts[a,2]>=.25 and pts[b,2]>=.25:
                cv2.line(frame,(int(pts[a,0]),int(pts[a,1])),
                         (int(pts[b,0]),int(pts[b,1])),(255,190,0),2)
        for px,py,confidence in pts:
            if confidence>=.25: cv2.circle(frame,(int(px),int(py)),3,(255,255,255),-1)
        label=f"ID {person.person_id} | {'FALL ALERT' if person.is_fallen else 'Normal'}"
        cv2.rectangle(frame,(x,max(0,y-29)),(x+min(w,230),y),color,-1)
        cv2.putText(frame,label,(x+5,max(18,y-9)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
        cv2.putText(frame,f"Score {person.posture_score:.2f}  Angle {person.torso_angle:.0f}",
                    (x,min(height-8,y+h+20)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
    alerts=sum(p.is_fallen for p in detections)
    cv2.rectangle(frame,(8,8),(390,43),(25,25,25),-1)
    cv2.putText(frame,f"People: {len(detections)} | Active fall alerts: {alerts}",(18,32),
                cv2.FONT_HERSHEY_SIMPLEX,.62,(0,0,255) if alerts else (80,240,100),2)
    return frame
