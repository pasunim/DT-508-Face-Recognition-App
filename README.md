# DT-508 Face Recognition App

แอปพลิเคชันจดจำใบหน้าแบบ Real-time ที่พัฒนาด้วย Python, OpenCV และ DeepFace โดยใช้กล้องเว็บแคมตรวจจับและระบุตัวตนของบุคคลจากฐานข้อมูลที่บันทึกไว้

---

## หลักการทำงาน

1. **สร้างฐานข้อมูล** — `setup_db.py` สร้างฐานข้อมูล SQLite (`face_recognition.db`) ที่เก็บชื่อและชื่อไฟล์รูปภาพของแต่ละบุคคล
2. **สร้าง Embedding** — เมื่อเริ่มแอป `load_faces.py` จะอ่านรูปภาพของแต่ละคนจากโฟลเดอร์ `images/` แล้วแปลงใบหน้าเป็นเวกเตอร์ขนาด 512 มิติ (Face Embedding) ด้วยโมเดล **Facenet512** ผ่าน DeepFace
3. **จดจำใบหน้าแบบ Real-time** — `main.py` จับภาพจากกล้อง โดยทุก ๆ 1 เฟรมเว้น 1 เฟรม จะส่งรูปไปให้ DeepFace ตรวจจับใบหน้าและสร้าง Embedding จากนั้นนำไปเปรียบเทียบกับ Embedding ของทุกคนในฐานข้อมูลด้วย **L2 Distance** (หลัง Normalize เวกเตอร์แล้ว) หากระยะห่างต่ำกว่า Threshold จะแสดงชื่อ มิเช่นนั้นจะแสดงว่า **UNKNOWN**

### ค่า Distance และ Threshold คืออะไร?

หลังจาก Normalize เวกเตอร์ทั้งสองให้มีขนาด 1 แล้ว ค่า L2 Distance จะอยู่ในช่วง **0** (หน้าเหมือนกันทุกประการ) ถึง **2** (แตกต่างกันอย่างสิ้นเชิง) ค่า Threshold เริ่มต้นคือ **0.6** — ยิ่งค่าน้อยยิ่งเข้มงวด สามารถปรับได้ใน `main.py` ตามสภาพแสงและคุณภาพของกล้อง

---

## ความต้องการของระบบ

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (Package Manager)

Dependencies (จัดการโดย uv):
- `deepface` — ตรวจจับใบหน้าและสร้าง Embedding
- `opencv-python` — จับภาพจากกล้องและวาดกรอบใบหน้า
- `numpy` — คำนวณระยะห่างระหว่างเวกเตอร์
- `tf-keras` — Backend ที่ DeepFace ต้องใช้

---

## การติดตั้ง

1. Clone โปรเจกต์:
   ```bash
   git clone https://github.com/pasunim/DT-508-Face-Recognition-App.git
   cd DT-508-Face-Recognition-App
   ```

2. ติดตั้ง Dependencies:
   ```bash
   uv sync
   ```

---

## วิธีใช้งาน

### ขั้นตอนที่ 1 — เพิ่มรูปภาพใบหน้า

วางรูปภาพอ้างอิงไว้ในโฟลเดอร์ `images/` แต่ละรูปควรมี **ใบหน้าตรงชัดเจนเพียงหนึ่งคน** และชื่อไฟล์ต้องตรงกับที่บันทึกในฐานข้อมูล

```
images/
├── somchai.jpg
├── malee.jpg
└── ...
```

### ขั้นตอนที่ 2 — แก้ไขข้อมูลบุคคล (ถ้าต้องการ)

เปิดไฟล์ `setup_db.py` แล้วแก้ไข `sample_data` ให้ตรงกับรูปภาพของคุณ:

```python
sample_data = [
    ("ชื่อ", "นามสกุล", "Male", "somchai.jpg"),
    ...
]
```

### ขั้นตอนที่ 3 — สร้างฐานข้อมูล

```bash
uv run setup_db.py
```

คำสั่งนี้จะสร้างไฟล์ `face_recognition.db` พร้อมข้อมูลบุคคลที่กำหนดไว้

### ขั้นตอนที่ 4 — รันแอป

```bash
uv run main.py
```

หน้าต่างกล้องจะเปิดขึ้น ใบหน้าที่ตรวจพบจะถูกล้อมด้วยกรอบสีเขียวและแสดงชื่อด้านล่าง หากจำไม่ได้จะแสดงว่า **UNKNOWN**

กด `q` เพื่อปิดแอป

---

## โครงสร้างโปรเจกต์

```
face-recognition-app/
├── images/              # รูปภาพอ้างอิงของแต่ละบุคคล
├── main.py              # แอปหลัก — วนลูปจับภาพ ตรวจจับ และจดจำใบหน้า
├── load_faces.py        # โหลดและสร้าง Embedding จากฐานข้อมูล
├── setup_db.py          # สร้างฐานข้อมูล SQLite และเพิ่มข้อมูลบุคคล
├── pyproject.toml       # กำหนด Dependencies ของโปรเจกต์
└── uv.lock              # Lock file สำหรับ Dependencies
```

---

## การปรับแต่ง Threshold

| ปัญหา | วิธีแก้ |
|-------|---------|
| คนที่รู้จักแต่แสดงเป็น UNKNOWN | เพิ่มค่า Threshold (เช่น `1.1`) หรือใช้รูปภาพที่แสงดีขึ้น |
| จำผิดคน | ลดค่า Threshold (เช่น `0.5`) |
| ประสิทธิภาพช้า | แอปประมวลผลทุก ๆ 2 เฟรมอยู่แล้วเพื่อลดโหลด |
| ความแม่นยำต่ำ | ใช้รูปภาพหลายรูปต่อคน แล้วนำ Embedding มาเฉลี่ยกัน |

---

## License

MIT License — Copyright (c) 2026 [Pasu Nimsuwan](https://github.com/pasunim)

ดูรายละเอียดเพิ่มเติมได้ที่ไฟล์ [LICENSE](LICENSE)

---

## โครงสร้างฐานข้อมูล

```sql
CREATE TABLE persons (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,  -- ชื่อ
    surname TEXT,           -- นามสกุล
    gender  TEXT,           -- เพศ
    image   TEXT            -- ชื่อไฟล์รูปใน images/ เช่น "somchai.jpg"
);
```
