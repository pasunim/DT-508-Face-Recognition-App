import cv2, numpy as np
from deepface import DeepFace
from load_faces import load_known_faces

person_face_encodings, person_face_names = load_known_faces()


def find_match(embedding, threshold=0.6):
    min_dist = float("inf")
    match_name = "UNKNOWN"
    emb = np.array(embedding)
    emb = emb / np.linalg.norm(emb)
    for known_enc, name in zip(person_face_encodings, person_face_names):
        known = np.array(known_enc)
        known = known / np.linalg.norm(known)
        dist = np.linalg.norm(emb - known)
        print(f"  dist to {name}: {dist:.4f}")
        if dist < min_dist:
            min_dist = dist
            match_name = name
    print(f"  => best match: {match_name} (dist={min_dist:.4f}, threshold={threshold})")
    return match_name if min_dist < threshold else "UNKNOWN"


videoCapture = cv2.VideoCapture(0)
frameProcess = True
data_locations, data_names = [], []
while True:
    ret, frame = videoCapture.read()
    if not ret:
        break
    resizing = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    if frameProcess:
        results = DeepFace.represent(
            img_path=resizing, model_name="Facenet512", enforce_detection=False
        )
        data_locations, data_names = [], []
        for face in results:
            r = face["facial_area"]
            x, y, w, h = r["x"] * 4, r["y"] * 4, r["w"] * 4, r["h"] * 4
            data_locations.append((x, y, w, h))
            data_names.append(find_match(face["embedding"]))
    frameProcess = not frameProcess

    for (x, y, w, h), name in zip(data_locations, data_names):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (26, 174, 10), 2)
        cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (26, 174, 10), cv2.FILLED)
        cv2.putText(
            frame,
            name,
            (x + 6, y + h - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            1,
        )
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()
