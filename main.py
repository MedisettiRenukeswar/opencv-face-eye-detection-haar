import os
import cv2


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_cascades():
    face_path = os.path.join("haar", "haarcascade_frontalface_default.xml")
    eye_path = os.path.join("haar", "haarcascade_eye.xml")

    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Missing: {face_path}")
    if not os.path.exists(eye_path):
        raise FileNotFoundError(f"Missing: {eye_path}")

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade.")
    if eye_cascade.empty():
        raise RuntimeError("Failed to load eye cascade.")

    return face_cascade, eye_cascade


def main():
    print("=== OpenCV Face + Eye Detection (Haar, Webcam) ===")

    outputs_dir = "outputs"
    ensure_dir(outputs_dir)

    try:
        face_cascade, eye_cascade = load_cascades()
    except Exception as e:
        print(f"❌ Error loading cascades: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Webcam opened.")
    print("Controls: 'q' → quit, 's' → save snapshot")

    snapshot_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from webcam.")
            break

        # Optional resize for speed
        # frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Draw over a copy
        annotated = frame.copy()

        face_count = 0

        for (x, y, w, h) in faces:
            face_count += 1

            # Draw face rectangle
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Label face
            label = f"Face #{face_count}"
            cv2.putText(
                annotated,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Eye detection inside face ROI
            face_roi_gray = gray[y:y + h, x:x + w]
            face_roi_color = annotated[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            eye_count = 0
            for (ex, ey, ew, eh) in eyes:
                eye_count += 1
                cv2.rectangle(
                    face_roi_color,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    (255, 0, 0),
                    2
                )
                cv2.putText(
                    face_roi_color,
                    f"eye",
                    (ex, ey - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Summary text
        summary = f"Faces: {face_count}"
        cv2.putText(
            annotated,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Face + Eye Detection (Haar)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_path = os.path.join(
                outputs_dir, f"webcam_faces_snapshot_{snapshot_idx}.jpg"
            )
            cv2.imwrite(out_path, annotated)
            print(f"📸 Saved snapshot to {out_path}")
            snapshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam released, windows closed.")


if __name__ == "__main__":
    main()
