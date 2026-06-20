import sqlite3
from deepface import DeepFace
import numpy as np


def load_known_faces():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, image FROM persons")
    rows = cursor.fetchall()
    conn.close()
    encodings, names = [], []

    for name, image_file in rows:
        result = DeepFace.represent(
            img_path=f"images/{image_file}",
            model_name="Facenet512",
            enforce_detection=False,
        )
        encodings.append(result[0]["embedding"])
        names.append(name)
    return encodings, names
