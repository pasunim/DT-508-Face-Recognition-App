import sqlite3


def create_database():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    surname TEXT,
    gender TEXT,
    image TEXT -- ชื่อไฟล์รูปภำพ เช่น "siwakorn.jpg"
    )
    """)

    sample_data = [
        ("Anutin", "Error", "Male", "anutin.jpg"),
        ("Siwakorn", "Banluesapy", "Male", "siwakorn.jpg"),
        ("Elon", "Musk", "Male", "elon.jpg"),
    ]

    cursor.executemany(
        """
    INSERT OR IGNORE INTO persons (name, surname, gender, image)
    VALUES (?, ?, ?, ?)
    """,
        sample_data,
    )

    conn.commit()
    conn.close()
    print("สร้างฐานข้อมูลและเพิ่มข้อมูลสำเร็จ")


if __name__ == "__main__":
    create_database()
