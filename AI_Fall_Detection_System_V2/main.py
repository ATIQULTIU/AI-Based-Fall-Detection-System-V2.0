
"""
AI Fall Detection System V2.0
Multi-person AI fall detection with YOLO pose tracking.

Developer: MD. Atiqul Islam (Atik)
Email: atik.cmttiu1001@gmail.com
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
from fall_detection.detector import FallDetectionSystem, draw_detections


# Developer Information
DEVELOPER_NAME = "MD. Atiqul Islam (Atik)"
DEVELOPER_EMAIL = "atik.cmttiu1001@gmail.com"


def parse_source(value):
    return int(value) if value.isdigit() else value


def main():
    p = argparse.ArgumentParser(
        description="Multi-person AI fall detection with YOLO pose tracking."
    )

    p.add_argument('--source', default='0',
                   help='Webcam index or video file path')
    p.add_argument('--model', default='yolo11n-pose.pt',
                   help='YOLO pose weights')
    p.add_argument('--conf', type=float, default=.35,
                   help='Detection confidence 0-1')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--confirm', type=float, default=.8,
                   help='Fall-like posture duration in seconds')
    p.add_argument('--angle', type=float, default=55.,
                   help='Torso angle threshold in degrees')
    p.add_argument('--horizontal-ratio', type=float, default=1.15)
    p.add_argument('--device', default=None,
                   help='Inference device, e.g. cpu or 0')
    p.add_argument('--output', default='',
                   help='Optional annotated MP4 output')
    p.add_argument('--log', default='output/fall_events.csv')
    p.add_argument('--no-display', action='store_true')

    args = p.parse_args()

    # Display application information
    print("=" * 60)
    print("       AI FALL DETECTION SYSTEM V2.0")
    print("=" * 60)
    print(f"Developer : {DEVELOPER_NAME}")
    print(f"Email     : {DEVELOPER_EMAIL}")
    print("=" * 60)

    cap = cv2.VideoCapture(parse_source(args.source))

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {args.source}")
        sys.exit(1)

    try:
        system = FallDetectionSystem(
            args.model,
            args.conf,
            args.imgsz,
            args.confirm,
            args.angle,
            args.horizontal_ratio,
            args.device
        )

    except Exception as exc:
        cap.release()
        print(f"ERROR initializing model: {exc}")
        print("Check dependencies/model access.")
        print("Default weights download on first run.")
        sys.exit(1)

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    new = not log.exists()
    logfile = log.open('a', newline='', encoding='utf-8')
    writer_csv = csv.writer(logfile)

    if new:
        writer_csv.writerow([
            'timestamp',
            'person_id',
            'event',
            'detection_confidence',
            'posture_score',
            'torso_angle_degrees',
            'bbox_aspect_ratio'
        ])

    out = Path(args.output) if args.output else None

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)

    video_writer = None

    print("AI Fall Detection System V2.0 started.")
    print("Press Q or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            people = system.process(frame)

            for person in people:
                if person.just_fell:
                    writer_csv.writerow([
                        datetime.now().isoformat(timespec='seconds'),
                        person.person_id,
                        'FALL_DETECTED',
                        f'{person.confidence:.3f}',
                        f'{person.posture_score:.3f}',
                        f'{person.torso_angle:.1f}',
                        f'{person.aspect_ratio:.2f}'
                    ])

                    logfile.flush()

            frame = draw_detections(frame, people)

            if out:
                if video_writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.
                    h, w = frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        str(out),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (w, h)
                    )

                    if not video_writer.isOpened():
                        raise RuntimeError(f'Cannot create {out}')

                video_writer.write(frame)

            if not args.no_display:
                cv2.imshow(
                    'AI Fall Detection System V2.0 | Q to quit',
                    frame
                )

                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

    except KeyboardInterrupt:
        print('\nStopped by user.')

    finally:
        cap.release()
        logfile.close()

        if video_writer:
            video_writer.release()

        if not args.no_display:
            cv2.destroyAllWindows()

    print(f'Fall event log: {log}')

    if out:
        print(f'Annotated video: {out}')

    print("=" * 60)
    print(f"Developed by: {DEVELOPER_NAME}")
    print(f"Contact     : {DEVELOPER_EMAIL}")
    print("=" * 60)


if __name__ == '__main__':
    main()
