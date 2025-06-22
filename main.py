import asyncio
import base64
import csv
import io
import json
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import google.generativeai as genai
import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
import supervision as sv
import torch
import uvicorn
from fastapi import (Body, Depends, FastAPI, File, HTTPException, Path, Query,
                     Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageFile
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                              TableStyle)
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash

ImageFile.LOAD_TRUNCATED_IMAGES = True
GEMINI_API_KEY = ""
MODEL_PATH = "./model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thread_pool = ThreadPoolExecutor(max_workers=8)

models = {
    "vehicle": {"model": None, "classes": None, "id2label": None},
    "type_plate": {"model": None, "classes": None, "id2label": None}
}
gemini_model = None

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'smart_parking'
}
connection_pool = None
def warmup_models():
    print("Memulai proses warm-up untuk model AI...")
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

    for model_type, model_data in models.items():
        if model_data.get("model"):
            print(f"Melakukan warm-up untuk model '{model_type}'...")
            try:
                model_data["model"].predict(dummy_image, verbose=False)
                print(f"Warm-up untuk model '{model_type}' berhasil.")
            except Exception as e:
                print(f"Gagal melakukan warm-up untuk model '{model_type}': {e}")
    print("Proses warm-up selesai.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection_pool
    print("Membuat direktori dan menginisialisasi pool koneksi database...")
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)

    try:
        connection_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="smartparking_pool",
            pool_size=10,
            pool_reset_session=True,
            **DB_CONFIG
        )
        print("Pool koneksi database berhasil dibuat.")
    except Error as e:
        print(f"Gagal membuat pool koneksi database: {e}")
        connection_pool = None

    print("Memuat model AI...")
    await asyncio.gather(
        asyncio.to_thread(ModelLoader.load_vehicle_model),
        asyncio.to_thread(ModelLoader.load_type_plate_model),
        asyncio.to_thread(ModelLoader.initialize_gemini)
    )
    print("Model AI berhasil dimuat.")
    
    await asyncio.to_thread(warmup_models)

    yield
    
    thread_pool.shutdown(wait=True)
    print("Aplikasi dimatikan.")

app = FastAPI(
    title="Smart Parking and Vehicle Detection API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_db_connection():
    if connection_pool is None:
        raise HTTPException(status_code=503, detail="Pool koneksi database tidak tersedia.")
    
    connection = None
    try:
        connection = connection_pool.get_connection()
        yield connection
    except Error as e:
        raise HTTPException(status_code=503, detail=f"Gagal mendapatkan koneksi dari pool: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()

class ModelLoader:
    @staticmethod
    def load_json(filepath: str) -> Dict:
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def load_vehicle_model():
        models["vehicle"]["classes"] = ModelLoader.load_json(f"{MODEL_PATH}/vehicle-plate-classes.json")
        models["vehicle"]["id2label"] = ModelLoader.load_json(f"{MODEL_PATH}/vehicle-plate-id2label.json")
        models["vehicle"]["model"] = YOLO(f"{MODEL_PATH}/vehicle-plate-detect.onnx")

    @staticmethod
    def load_type_plate_model():
        models["type_plate"]["classes"] = ModelLoader.load_json(f"{MODEL_PATH}/type-plate-classes.json")
        models["type_plate"]["id2label"] = ModelLoader.load_json(f"{MODEL_PATH}/type-plate-id2label.json")
        models["type_plate"]["model"] = YOLO(f"{MODEL_PATH}/type-plate-detect.onnx")

    @staticmethod
    def initialize_gemini():
        global gemini_model
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
        except Exception as e:
            print(f"Gagal menginisialisasi Gemini: {e}")
            gemini_model = None

class ImageProcessor:
    @staticmethod
    def safe_load_image(image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            _ = np.array(image)
            return image
        except Exception:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_file.flush()
                image = Image.open(tmp_file.name)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                os.unlink(tmp_file.name)
                return image
            except Exception as e2:
                try:
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image_cv is None:
                        raise ValueError("OpenCV could not decode image")
                    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(image_rgb)
                except Exception as e3:
                    raise HTTPException(status_code=400, detail=f"Could not load image: {str(e3)}")

    @staticmethod
    def preprocess_plate(plate_image: np.ndarray) -> Image.Image:
        try:
            plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            original_height, original_width = plate_rgb.shape[:2]
            target_width = max(400, original_width * 2)
            scale = target_width / original_width
            target_height = int(original_height * scale)
            interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            plate_resized = cv2.resize(plate_rgb, (target_width, target_height), interpolation=interpolation)
            gray = cv2.cvtColor(plate_resized, cv2.COLOR_RGB2GRAY)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            kernel_sharpen = np.array([[-1,-1,-1,-1,-1], [-1,2,2,2,-1], [-1,2,8,2,-1], [-1,2,2,2,-1], [-1,-1,-1,-1,-1]]) / 8.0
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morphed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
            final = cv2.GaussianBlur(morphed, (1, 1), 0)
            final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(final_rgb)
        except Exception:
            return Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

class DetectionProcessor:
    @staticmethod
    def detect_objects(image: Image.Image, model_type: str, confidence: float = 0.3) -> Tuple[List[Dict], sv.Detections]:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            model_data = models[model_type]
            model, classes, id2label = model_data["model"], model_data["classes"], model_data["id2label"]
            results = model.predict(image, conf=confidence, verbose=True)[0]
            
            result_path = os.path.join('static', 'results', f"{model_type}_result_{uuid.uuid4().hex[:8]}.jpg")
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            results.save(filename=result_path)
            
            detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.5)
            detection_results = []
            for idx in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[idx]
                confidence_score = detections.confidence[idx]
                class_id = detections.class_id[idx]
                class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                class_name_mapped = id2label.get(str(class_id), class_name)
                
                detection_results.append({
                    "model": f"{model_type}-detect",
                    "class_id": int(class_id),
                    "class_name": class_name_mapped,
                    "confidence": float(confidence_score),
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                })
            return detection_results, detections
        except Exception as e:
            raise e

    @staticmethod
    async def detect_async(image: Image.Image, model_type: str, confidence: float = 0.3):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(thread_pool, DetectionProcessor.detect_objects, image, model_type, confidence)

class OCRProcessor:
    @staticmethod
    def extract_plate_regions(image: Image.Image, detections: List[Dict]) -> List[Dict]:
        plate_regions = []
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for detection in detections:
            if any(keyword in detection['class_name'].lower() for keyword in ['plat', 'plate', 'license']):
                bbox = detection['bbox']
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                padding = 20
                h, w = image_cv.shape[:2]
                x1_padded, y1_padded = max(0, x1 - padding), max(0, y1 - padding)
                x2_padded, y2_padded = min(w, x2 + padding), min(h, y2 + padding)
                cropped_plate = image_cv[y1_padded:y2_padded, x1_padded:x2_padded]
                if cropped_plate.size > 0:
                    plate_regions.append({
                        'image': cropped_plate, 'bbox': (x1, y1, x2, y2),
                        'confidence': detection['confidence'], 'model_source': detection['model'],
                        'class_name': detection['class_name']
                    })
        return plate_regions

    @staticmethod
    async def ocr_with_gemini(plate_image: Image.Image) -> Dict:
        if gemini_model is None:
            return {'text': 'Gemini API not available', 'status': 'error'}
        try:
            prompt = """Anda adalah expert pembaca plat nomor Indonesia. Analisis gambar dan berikan HANYA teks plat nomor (contoh: B 1234 ABC). Jika tidak terbaca, jawab: "TIDAK_TERBACA"."""
            def run_ocr():
                response = gemini_model.generate_content([prompt, plate_image])
                if response and response.text:
                    ocr_text = ' '.join(response.text.strip().split())
                    is_success = ocr_text != "TIDAK_TERBACA"
                    return {'text': ocr_text if is_success else "Tidak dapat dibaca", 'status': 'success' if is_success else 'failed'}
                return {'text': 'No response from Gemini', 'status': 'error'}
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(thread_pool, run_ocr)
        except Exception:
            return {'text': 'OCR Error', 'status': 'error'}

    @staticmethod
    async def process_plates(image: Image.Image, all_detections: List[Dict]) -> List[Dict]:
        plate_regions = OCRProcessor.extract_plate_regions(image, all_detections)
        if not plate_regions:
            return []
        
        async def process_single_plate(i: int, plate_region: Dict) -> Dict:
            loop = asyncio.get_event_loop()
            processed_plate = await loop.run_in_executor(thread_pool, ImageProcessor.preprocess_plate, plate_region['image'])
            ocr_result = await OCRProcessor.ocr_with_gemini(processed_plate)
            ocr_result.update({
                'bbox': plate_region['bbox'], 'detection_confidence': plate_region['confidence'],
                'plate_number': i + 1, 'model_source': plate_region['model_source'],
                'detected_class': plate_region['class_name']
            })
            return ocr_result
        
        ocr_tasks = [process_single_plate(i, r) for i, r in enumerate(plate_regions)]
        return await asyncio.gather(*ocr_tasks)

class AnnotationProcessor:
    @staticmethod
    def annotate_combined(image: Image.Image, vehicle_detections: sv.Detections,
                          type_plate_detections: sv.Detections, ocr_results: Optional[List[Dict]] = None) -> Image.Image:
        try:
            image_np = np.array(image)
            annotators = {
                "vehicle": (sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN), sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=sv.Color.GREEN)),
                "type_plate": (sv.BoxAnnotator(thickness=2, color=sv.Color.BLUE), sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=sv.Color.BLUE))
            }

            for model_type, detections in [("vehicle", vehicle_detections), ("type_plate", type_plate_detections)]:
                if len(detections.xyxy) > 0:
                    labels = []
                    for idx, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
                        class_name = models[model_type]["id2label"].get(str(class_id), f"class_{class_id}")
                        label = f"[{model_type.capitalize()}] {class_name}: {conf:.2f}"
                        if any(k in class_name.lower() for k in ['plat', 'plate']) and ocr_results:
                            x1, y1, x2, y2 = detections.xyxy[idx]
                            matched_ocr = min(
                                (ocr for ocr in ocr_results if ocr.get('model_source') == f'{model_type}-detect'),
                                key=lambda ocr: abs(x1 - ocr['bbox'][0]) + abs(y1 - ocr['bbox'][1]), default=None
                            )
                            if matched_ocr and matched_ocr['status'] == 'success':
                                label = f"[{model_type.capitalize()}] {matched_ocr['text']}"
                        labels.append(label)
                    
                    box_annotator, label_annotator = annotators[model_type]
                    image_np = box_annotator.annotate(image_np, detections)
                    image_np = label_annotator.annotate(image_np, detections, labels=labels)
            
            return Image.fromarray(image_np)
        except Exception:
            return image

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class UserUpdate(BaseModel):
    username: str
    role: str
    password: Optional[str] = None

class KapasitasUpdate(BaseModel):
    jenis_kendaraan: str
    kapasitas_total: int

class VehicleEntry(BaseModel):
    nomor_plat: str
    tanggal_masuk: str
    jenis_kendaraan: Optional[str] = 'Tidak Diketahui'
    tipe_plat: Optional[str] = 'Sipil'
    status: Optional[str] = 'Aktif'

class KendaraanKeluar(BaseModel):
    id_masuk: int

class AksesItem(BaseModel):
    nomor_plat: str
    status: str
    
def save_detection_to_db(conn: mysql.connector.MySQLConnection, nomor_plat: str, jenis_kendaraan: str, tipe_plat: str):
    if not nomor_plat or not jenis_kendaraan:
        return {"status": "error", "message": "Nomor plat atau jenis kendaraan tidak valid untuk disimpan."}

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id_masuk FROM kendaraan_masuk WHERE nomor_plat = %s AND status = 'Aktif' LIMIT 1", (nomor_plat,))
        if cursor.fetchone():
            return {"status": "already_exists", "message": f"Kendaraan {nomor_plat} sudah ada dan aktif."}
        
        waktu_masuk_db = datetime.now()
        cursor.execute(
            "INSERT INTO kendaraan_masuk (nomor_plat, jenis_kendaraan, tanggal_masuk, tipe_plat, status) VALUES (%s, %s, %s, %s, 'Aktif')",
            (nomor_plat, jenis_kendaraan, waktu_masuk_db, tipe_plat)
        )
        conn.commit()
        return {"status": "success", "message": f"Kendaraan {nomor_plat} berhasil disimpan ke database."}
    except Error as err:
        conn.rollback()
        return {"status": "db_error", "message": f"Gagal menyimpan data: {err}"}
    finally:
        if cursor: cursor.close()

def generate_pdf_report(data_laporan: List[Dict]) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = [Paragraph("Laporan Transaksi Parkir", styles['h1']), Spacer(1, 24)]
    
    if data_laporan:
        headers = ["Nomor Plat", "Jenis", "Tipe Plat", "Tgl Masuk", "Tgl Keluar", "Durasi (m)", "Biaya"]
        table_data = [headers]
        for item in data_laporan:
            biaya_str = "Rp {:,.0f}".format(item.get('biaya', 0) or 0)
            table_data.append([
                item.get('nomor_plat', '-'), item.get('jenis_kendaraan', '-'),
                item.get('tipe_plat', '-'), item.get('tanggal_masuk_formatted', '-'),
                item.get('tanggal_keluar_formatted', '-'), str(item.get('durasi', 0) or 0), biaya_str
            ])
        table = Table(table_data, colWidths=[120, 60, 70, 100, 100, 60, 90])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 10), ('BACKGROUND',(0,1),(-1,-1),colors.aliceblue),
            ('GRID',(0,0),(-1,-1),1,colors.black), ('FONTSIZE', (0,1), (-1,-1), 8)
        ]))
        story.append(table)
    else:
        story.append(Paragraph("Tidak ada data laporan untuk ditampilkan.", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def hitung_biaya_parkir(conn: mysql.connector.MySQLConnection, id_masuk: int, waktu_keluar: datetime):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT nomor_plat, tanggal_masuk, jenis_kendaraan, tipe_plat FROM kendaraan_masuk WHERE id_masuk = %s", (id_masuk,))
        kendaraan = cursor.fetchone()
        if not kendaraan: return None
        
        cursor.execute("SELECT status FROM akses WHERE nomor_plat = %s AND status = 'WHITELIST'", (kendaraan['nomor_plat'],))
        whitelist_status = cursor.fetchone()

        waktu_masuk = kendaraan['tanggal_masuk']
        durasi_menit = round((waktu_keluar - waktu_masuk).total_seconds() / 60)

        if whitelist_status:
            return {"durasi_menit": durasi_menit, "biaya_rp": 0, "waktu_masuk": waktu_masuk.isoformat()}

        cursor.execute("SELECT setting_name, setting_value FROM pengaturan_sistem")
        pengaturan_raw = cursor.fetchall()
        pengaturan = {item['setting_name']: int(item['setting_value']) if item['setting_value'].isdigit() else item['setting_value'] for item in pengaturan_raw}

        jenis = kendaraan['jenis_kendaraan'].lower()
        tarif_pertama = pengaturan.get(f'tarif_pertama_{jenis}', 0)
        tarif_perjam = pengaturan.get(f'tarif_perjam_{jenis}', 0)
        tarif_maksimum = pengaturan.get(f'tarif_maksimum_harian_{jenis}', float('inf'))
        toleransi_menit = pengaturan.get('toleransi_masuk_menit', 0)

        if durasi_menit <= toleransi_menit:
            return {"durasi_menit": durasi_menit, "biaya_rp": 0, "waktu_masuk": waktu_masuk.isoformat()}
        
        jumlah_hari = durasi_menit // 1440
        sisa_menit = durasi_menit % 1440
        
        biaya_sisa = 0
        if sisa_menit > toleransi_menit:
            sisa_menit_efektif = sisa_menit - toleransi_menit
            if sisa_menit_efektif <= 60:
                biaya_sisa = tarif_pertama
            else:
                jam_berikutnya = -(- (sisa_menit_efektif - 60) // 60)
                biaya_sisa = tarif_pertama + (jam_berikutnya * tarif_perjam)
            
            biaya_sisa = min(biaya_sisa, tarif_maksimum)

        biaya_final = (jumlah_hari * tarif_maksimum) + biaya_sisa
        
        return {"durasi_menit": durasi_menit, "biaya_rp": biaya_final, "waktu_masuk": waktu_masuk.isoformat()}
    except (Error, ValueError, TypeError) as e:
        print(f"Error calculating cost: {e}")
        return None
    finally:
        if cursor: cursor.close()

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/api/dashboard_stats')
def get_dashboard_stats(
    end_date_req: Optional[str] = Query(None, alias='end_date'),
    start_date_req: Optional[str] = Query(None, alias='start_date'),
    conn: mysql.connector.MySQLConnection = Depends(get_db_connection)
):
    end_date_str = end_date_req if end_date_req else date.today().isoformat()
    start_date_str = start_date_req if start_date_req else (date.today() - timedelta(days=6)).isoformat()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT
                COALESCE(SUM(t.biaya), 0) AS total_pendapatan,
                COALESCE(COUNT(t.id_transaksi), 0) AS total_kendaraan,
                COALESCE(AVG(TIMESTAMPDIFF(MINUTE, km.tanggal_masuk, kk.tanggal_keluar)), 0) AS avg_duration_minutes
            FROM transaksi t
            INNER JOIN kendaraan_masuk km ON t.id_masuk = km.id_masuk
            INNER JOIN kendaraan_keluar kk ON t.id_keluar = kk.id_keluar
            WHERE DATE(kk.tanggal_keluar) BETWEEN %s AND %s
        """, (start_date_str, end_date_str))
        stat_cards = cursor.fetchone()
        if stat_cards is None:
            stat_cards = {'total_pendapatan': 0, 'total_kendaraan': 0, 'avg_duration_minutes': 0}
        stat_cards['avg_duration_minutes'] = round(stat_cards.get('avg_duration_minutes', 0) or 0)

        cursor.execute("""
            SELECT DATE(kk.tanggal_keluar) AS tanggal, SUM(t.biaya) AS pendapatan
            FROM transaksi t
            JOIN kendaraan_keluar kk ON t.id_keluar = kk.id_keluar
            WHERE DATE(kk.tanggal_keluar) BETWEEN %s AND %s
            GROUP BY DATE(kk.tanggal_keluar) ORDER BY tanggal ASC
        """, (start_date_str, end_date_str))
        pendapatan_harian = cursor.fetchall()
        for item in pendapatan_harian:
            if isinstance(item.get('tanggal'), date):
                item['tanggal'] = item['tanggal'].strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT HOUR(km.tanggal_masuk) AS jam, COUNT(km.id_masuk) AS jumlah
            FROM kendaraan_masuk km
            WHERE DATE(km.tanggal_masuk) BETWEEN %s AND %s
            AND km.jenis_kendaraan IN (SELECT DISTINCT k.jenis_kendaraan FROM kapasitas k)
            GROUP BY HOUR(km.tanggal_masuk) ORDER BY jam ASC
        """, (start_date_str, end_date_str))
        arus_per_jam = cursor.fetchall()

        cursor.execute("""
            SELECT km.jenis_kendaraan, COUNT(km.id_masuk) AS jumlah
            FROM kendaraan_masuk km
            WHERE DATE(km.tanggal_masuk) BETWEEN %s AND %s
            AND km.jenis_kendaraan IN (SELECT DISTINCT k.jenis_kendaraan FROM kapasitas k)
            GROUP BY km.jenis_kendaraan
        """, (start_date_str, end_date_str))
        distribusi_kendaraan = cursor.fetchall()

        cursor.execute("""
            SELECT km.jenis_kendaraan, COALESCE(SUM(t.biaya), 0) AS total_pendapatan
            FROM transaksi t
            JOIN kendaraan_masuk km ON t.id_masuk = km.id_masuk
            JOIN kendaraan_keluar kk ON t.id_keluar = kk.id_keluar
            WHERE DATE(kk.tanggal_keluar) BETWEEN %s AND %s
            GROUP BY km.jenis_kendaraan
        """, (start_date_str, end_date_str))
        pendapatan_per_jenis = cursor.fetchall()

        return {
            'stat_cards': stat_cards,
            'pendapatan_harian': pendapatan_harian,
            'arus_per_jam': arus_per_jam,
            'distribusi_kendaraan': distribusi_kendaraan,
            'pendapatan_per_jenis': pendapatan_per_jenis
        }
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.post('/api/login')
def login(data: UserLogin, conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True, buffered=True)
    try:
        cursor.execute("SELECT id_user, username, password, role FROM user WHERE username = %s", (data.username,))
        user_from_db = cursor.fetchone()
        if user_from_db and check_password_hash(user_from_db['password'], data.password):
            role = user_from_db.get('role', 'operator').strip().lower()
            avatar = user_from_db['username'][0].upper()
            return {'success': True, 'user': {'name': user_from_db['username'], 'avatar': avatar, 'role': role}}
        else:
            raise HTTPException(status_code=401, detail={'success': False, 'error': 'Username atau password salah'})
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id_user, username, password, role FROM user WHERE username = %s", (data.username,))
        user_from_db = cursor.fetchone()
        if user_from_db and check_password_hash(user_from_db['password'], data.password):
            role = user_from_db.get('role', 'operator').strip().lower()
            avatar = user_from_db['username'][0].upper()
            return {'success': True, 'user': {'name': user_from_db['username'], 'avatar': avatar, 'role': role}}
        else:
            raise HTTPException(status_code=401, detail={'success': False, 'error': 'Username atau password salah'})
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.get('/api/users')
def get_users(conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id_user, username, role FROM user ORDER BY username ASC")
        return cursor.fetchall()
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.post('/api/users', status_code=201)
def add_user(data: UserCreate, conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    hashed_password = generate_password_hash(data.password)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO user (username, password, role) VALUES (%s, %s, %s)", (data.username, hashed_password, data.role))
        conn.commit()
        return {'success': True}
    except Error as e:
        conn.rollback()
        if 'Duplicate entry' in str(e):
            raise HTTPException(status_code=409, detail='Username sudah ada')
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.put('/api/users/{id_user}')
def update_user(id_user: int, data: UserUpdate, conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor()
    try:
        if data.password and data.password.strip():
            hashed_password = generate_password_hash(data.password)
            cursor.execute("UPDATE user SET username = %s, role = %s, password = %s WHERE id_user = %s",
                           (data.username, data.role, hashed_password, id_user))
        else:
            cursor.execute("UPDATE user SET username = %s, role = %s WHERE id_user = %s",
                           (data.username, data.role, id_user))
        conn.commit()
        return {'success': True}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.delete('/api/users/{id_user}')
def delete_user(id_user: int, conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT role FROM user WHERE id_user = %s", (id_user,))
        user = cursor.fetchone()
        if user and user['role'].lower() == 'admin':
            cursor.execute("SELECT COUNT(*) as admin_count FROM user WHERE role = 'admin'")
            if cursor.fetchone()['admin_count'] <= 1:
                raise HTTPException(status_code=403, detail='Tidak dapat menghapus satu-satunya admin')
        
        cursor.execute("DELETE FROM user WHERE id_user = %s", (id_user,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail='User tidak ditemukan')
        return {'success': True}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.get('/api/monitoring')
def get_monitoring_data(conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT
                k.jenis_kendaraan, k.kapasitas_total,
                (SELECT COUNT(*) FROM kendaraan_masuk km WHERE km.jenis_kendaraan = k.jenis_kendaraan AND km.status = 'Aktif') AS kapasitas_terisi
            FROM kapasitas k
        """)
        kapasitas = cursor.fetchall()
        cursor.execute("""
            SELECT nomor_plat, jenis_kendaraan, tipe_plat, tanggal_masuk, status
            FROM kendaraan_masuk WHERE status = 'Aktif' ORDER BY tanggal_masuk DESC LIMIT 5
        """)
        recent_activities = cursor.fetchall()
        for activity in recent_activities:
            if isinstance(activity.get('tanggal_masuk'), datetime):
                activity['tanggal_masuk'] = activity['tanggal_masuk'].isoformat()
        return {'kapasitas': kapasitas, 'recent_activities': recent_activities}
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.get('/api/laporan')
def get_laporan_data(
    start_date_str: Optional[str] = Query(None, alias='start_date'),
    end_date_str: Optional[str] = Query(None, alias='end_date'),
    jenis_kendaraan_filter: Optional[str] = Query(None, alias='jenis_kendaraan'),
    tipe_plat_filter: Optional[str] = Query(None, alias='tipe_plat'),
    conn: mysql.connector.MySQLConnection = Depends(get_db_connection)
):
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT t.id_transaksi, km.nomor_plat, km.jenis_kendaraan, km.tipe_plat,
                   km.tanggal_masuk, kk.tanggal_keluar,
                   TIMESTAMPDIFF(MINUTE, km.tanggal_masuk, kk.tanggal_keluar) AS durasi, t.biaya
            FROM transaksi t
            JOIN kendaraan_masuk km ON t.id_masuk = km.id_masuk
            JOIN kendaraan_keluar kk ON t.id_keluar = kk.id_keluar
        """
        conditions, params = [], []
        if start_date_str:
            conditions.append("DATE(kk.tanggal_keluar) >= %s"); params.append(start_date_str)
        if end_date_str:
            conditions.append("DATE(kk.tanggal_keluar) <= %s"); params.append(end_date_str)
        if jenis_kendaraan_filter:
            conditions.append("km.jenis_kendaraan = %s"); params.append(jenis_kendaraan_filter)
        if tipe_plat_filter:
            conditions.append("km.tipe_plat = %s"); params.append(tipe_plat_filter)
        
        if conditions: query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY kk.tanggal_keluar DESC"
        
        cursor.execute(query, tuple(params))
        laporan_data = cursor.fetchall()
        for item in laporan_data:
            if isinstance(item.get('tanggal_masuk'), datetime):
                item['tanggal_masuk'] = item['tanggal_masuk'].isoformat()
            if isinstance(item.get('tanggal_keluar'), datetime):
                item['tanggal_keluar'] = item['tanggal_keluar'].isoformat()
            item['durasi'] = int(item.get('durasi') or 0)

        return {'laporan_data': laporan_data}
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.post('/api/kapasitas/update')
def update_kapasitas_parkir(data: KapasitasUpdate, conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    try:
        kapasitas_total_baru = int(data.kapasitas_total)
        if kapasitas_total_baru < 0:
            raise HTTPException(status_code=400, detail='Kapasitas total tidak boleh negatif')
    except ValueError:
        raise HTTPException(status_code=400, detail='Kapasitas total harus berupa angka')
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM kapasitas WHERE jenis_kendaraan = %s", (data.jenis_kendaraan,))
        if cursor.fetchone()[0] == 0:
            raise HTTPException(status_code=404, detail=f'Jenis kendaraan {data.jenis_kendaraan} tidak ditemukan')
        
        cursor.execute("UPDATE kapasitas SET kapasitas_total = %s WHERE jenis_kendaraan = %s",
                       (kapasitas_total_baru, data.jenis_kendaraan))
        conn.commit()
        
        if cursor.rowcount > 0:
            return {'success': True, 'message': f'Kapasitas {data.jenis_kendaraan} berhasil diperbarui.'}
        else:
            return {'success': True, 'message': 'Nilai kapasitas sama dengan sebelumnya, tidak ada perubahan.'}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.get('/api/laporan/pdf')
def download_laporan_pdf(
    start_date_str: Optional[str] = Query(None, alias='start_date'),
    end_date_str: Optional[str] = Query(None, alias='end_date'),
    jenis_kendaraan_filter: Optional[str] = Query(None, alias='jenis_kendaraan'),
    tipe_plat_filter: Optional[str] = Query(None, alias='tipe_plat'),
    conn: mysql.connector.MySQLConnection = Depends(get_db_connection)
):
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT km.nomor_plat, km.jenis_kendaraan, km.tipe_plat,
                   DATE_FORMAT(km.tanggal_masuk, '%Y-%m-%d %H:%i') AS tanggal_masuk_formatted,
                   DATE_FORMAT(kk.tanggal_keluar, '%Y-%m-%d %H:%i') AS tanggal_keluar_formatted,
                   TIMESTAMPDIFF(MINUTE, km.tanggal_masuk, kk.tanggal_keluar) AS durasi, t.biaya
            FROM transaksi t JOIN kendaraan_masuk km ON t.id_masuk = km.id_masuk
            JOIN kendaraan_keluar kk ON t.id_keluar = kk.id_keluar
        """
        conditions, params = [], []
        if start_date_str: conditions.append("DATE(kk.tanggal_keluar) >= %s"); params.append(start_date_str)
        if end_date_str: conditions.append("DATE(kk.tanggal_keluar) <= %s"); params.append(end_date_str)
        if jenis_kendaraan_filter: conditions.append("km.jenis_kendaraan = %s"); params.append(jenis_kendaraan_filter)
        if tipe_plat_filter: conditions.append("km.tipe_plat = %s"); params.append(tipe_plat_filter)
        if conditions: query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY kk.tanggal_keluar DESC"
        
        cursor.execute(query, tuple(params))
        data_laporan = cursor.fetchall()
        
        pdf_buffer = generate_pdf_report(data_laporan)
        filename = f'Laporan_Transaksi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        return StreamingResponse(pdf_buffer, media_type='application/pdf', headers={'Content-Disposition': f'attachment; filename="{filename}"'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Gagal membuat PDF: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.get('/api/laporan/csv')
def download_laporan_csv(
    start_date_str: Optional[str] = Query(None, alias='start_date'),
    end_date_str: Optional[str] = Query(None, alias='end_date'),
    jenis_kendaraan_filter: Optional[str] = Query(None, alias='jenis_kendaraan'),
    tipe_plat_filter: Optional[str] = Query(None, alias='tipe_plat'),
    conn: mysql.connector.MySQLConnection = Depends(get_db_connection)
):
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT km.nomor_plat, km.jenis_kendaraan, km.tipe_plat,
                   DATE_FORMAT(km.tanggal_masuk, '%Y-%m-%d %H:%i') AS tanggal_masuk,
                   DATE_FORMAT(kk.tanggal_keluar, '%Y-%m-%d %H:%i') AS tanggal_keluar,
                   TIMESTAMPDIFF(MINUTE, km.tanggal_masuk, kk.tanggal_keluar) AS durasi, t.biaya
            FROM transaksi t JOIN kendaraan_masuk km ON t.id_masuk = km.id_masuk
            JOIN kendaraan_keluar kk ON t.id_keluar = kk.id_keluar
        """
        conditions, params = [], []
        if start_date_str: conditions.append("DATE(kk.tanggal_keluar) >= %s"); params.append(start_date_str)
        if end_date_str: conditions.append("DATE(kk.tanggal_keluar) <= %s"); params.append(end_date_str)
        if jenis_kendaraan_filter: conditions.append("km.jenis_kendaraan = %s"); params.append(jenis_kendaraan_filter)
        if tipe_plat_filter: conditions.append("km.tipe_plat = %s"); params.append(tipe_plat_filter)
        if conditions: query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY kk.tanggal_keluar DESC"

        cursor.execute(query, tuple(params))
        laporan_data = cursor.fetchall()
        if not laporan_data:
            raise HTTPException(status_code=404, detail='Tidak ada data untuk diekspor')

        si = StringIO()
        cw = csv.writer(si, delimiter=';')
        headers = ["Nomor Plat", "Jenis Kendaraan", "Tipe Plat", "Tanggal Masuk", "Tanggal Keluar", "Durasi (Menit)", "Biaya (Rp)"]
        cw.writerow(headers)
        for row in laporan_data:
            cw.writerow([
                row.get('nomor_plat', ''), row.get('jenis_kendaraan', ''), row.get('tipe_plat', ''),
                row.get('tanggal_masuk', ''), row.get('tanggal_keluar', ''),
                row.get('durasi', 0), row.get('biaya', 0)
            ])
        output = BytesIO(si.getvalue().encode('utf-8-sig'))
        
        filename = f'Laporan_Transaksi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return StreamingResponse(output, media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="{filename}"'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Gagal membuat file CSV: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.get('/api/kendaraan/cari/{nomor_plat}')
def cari_kendaraan_aktif(nomor_plat: str = Path(..., title="Nomor Plat Kendaraan"), conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT id_masuk, nomor_plat, jenis_kendaraan, tipe_plat, tanggal_masuk FROM kendaraan_masuk
            WHERE nomor_plat = %s AND status = 'Aktif' ORDER BY tanggal_masuk DESC LIMIT 1
        """, (nomor_plat,))
        kendaraan = cursor.fetchone()
        
        if not kendaraan:
            raise HTTPException(status_code=404, detail=f'Kendaraan aktif dengan plat "{nomor_plat}" tidak ditemukan.')
        
        if isinstance(kendaraan.get('tanggal_masuk'), datetime):
            kendaraan['tanggal_masuk'] = kendaraan['tanggal_masuk'].isoformat()
        
        return kendaraan
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.get("/api/jenis_kendaraan", response_model=List[str])
def get_jenis_kendaraan(conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT DISTINCT jenis_kendaraan FROM kapasitas ORDER BY jenis_kendaraan ASC")
        return [row['jenis_kendaraan'] for row in cursor.fetchall()]
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.post("/api/detect/entry")
async def detect_vehicles_entry(
    file: UploadFile = File(...), 
    confidence: float = 0.3, 
    return_image: bool = True, 
    enable_ocr: bool = True,
    conn: mysql.connector.MySQLConnection = Depends(get_db_connection)
):
    try:
        image_bytes = await file.read()
        image = ImageProcessor.safe_load_image(image_bytes)
        vehicle_task = DetectionProcessor.detect_async(image, "vehicle", confidence)
        type_plate_task = DetectionProcessor.detect_async(image, "type_plate", confidence)
        (vehicle_results, v_dets), (type_plate_results, tp_dets) = await asyncio.gather(vehicle_task, type_plate_task)

        all_detections = vehicle_results + type_plate_results
        ocr_results = []
        db_save_status = {"status": "not_attempted", "message": "Penyimpanan tidak dilakukan."}

        plate_detected = any(any(k in d['class_name'].lower() for k in ['plat', 'plate']) for d in all_detections)
        
        if enable_ocr and gemini_model and plate_detected:
            ocr_results = await OCRProcessor.process_plates(image, all_detections)

            if ocr_results and ocr_results[0].get('status') == 'success':
                nomor_plat_terdeteksi = ocr_results[0]['text'].strip().upper()
                
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT status FROM akses WHERE nomor_plat = %s", (nomor_plat_terdeteksi,))
                akses_data = cursor.fetchone()
                
                if akses_data and akses_data['status'] == 'BLACKLIST':
                    raise HTTPException(status_code=403, detail=f"AKSES DITOLAK: {nomor_plat_terdeteksi} ada di daftar hitam.")
                
                jenis_kendaraan = next((d['class_name'] for d in vehicle_results if 'plat' not in d['class_name'].lower()), "Tidak Diketahui")
                tipe_plat = type_plate_results[0]['class_name'] if type_plate_results else "Sipil"

                # --- Blok Pengecekan Kapasitas Parkir ---
                # Query untuk mendapatkan kapasitas total dan terisi untuk jenis kendaraan yang terdeteksi
                cursor.execute("""
                    SELECT
                        k.kapasitas_total,
                        (SELECT COUNT(*) FROM kendaraan_masuk km WHERE km.jenis_kendaraan = k.jenis_kendaraan AND km.status = 'Aktif') AS kapasitas_terisi
                    FROM kapasitas k
                    WHERE k.jenis_kendaraan = %s
                """, (jenis_kendaraan,))
                kapasitas_info = cursor.fetchone()

                # Periksa apakah jenis kendaraan ada di tabel kapasitas dan apakah sudah penuh
                if kapasitas_info:
                    if kapasitas_info['kapasitas_terisi'] >= kapasitas_info['kapasitas_total']:
                        # Jika penuh, kirim error 409 Conflict dan hentikan proses
                        raise HTTPException(
                            status_code=409, 
                            detail=f"KAPASITAS PENUH: Parkir untuk '{jenis_kendaraan}' sudah penuh."
                        )
                # --- Akhir Blok Pengecekan Kapasitas ---
                
                # Jika kapasitas tersedia, lanjutkan untuk menyimpan data
                db_save_status = save_detection_to_db(conn, nomor_plat=nomor_plat_terdeteksi, jenis_kendaraan=jenis_kendaraan, tipe_plat=tipe_plat)
                
                cursor.close()

            elif ocr_results:
                db_save_status = {"status": "ocr_failed", "message": "Penyimpanan gagal karena OCR tidak dapat membaca plat."}

        response = {
            "filename": file.filename,
            "vehicle_detections": {"results": vehicle_results, "count": len(vehicle_results)},
            "type_plate_detections": {"results": type_plate_results, "count": len(type_plate_results)},
            "ocr_results": ocr_results,
            "database_status": db_save_status
        }

        if return_image:
            annotated_image = AnnotationProcessor.annotate_combined(image, v_dets, tp_dets, ocr_results)
            response["annotated_image"] = ImageProcessor.to_base64(annotated_image)
            
        return response
    
    except HTTPException as e:
        # Re-raise HTTPException agar FastAPI dapat menanganinya dengan benar
        raise e
    except Exception as e:
        print(f"Error in /api/detect/entry: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/api/info")
async def get_info():
    return {
        "vehicle_model_classes": len(models["vehicle"]["classes"] or []),
        "type_plate_model_classes": len(models["type_plate"]["classes"] or []),
        "device": str(DEVICE),
        "gemini_available": gemini_model is not None,
        "database_pool_available": connection_pool is not None,
        "status": "ready"
    }

@app.post("/api/detect/exit")
async def detect_and_exit_vehicle(
    file: UploadFile = File(...), 
    confidence: float = 0.3,
    conn: mysql.connector.MySQLConnection = Depends(get_db_connection)
):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")
        image_bytes = await file.read()
        image = ImageProcessor.safe_load_image(image_bytes)
        vehicle_results, vehicle_detections = await DetectionProcessor.detect_async(image, "vehicle", confidence)
        
        ocr_results = []
        if any(any(k in d['class_name'].lower() for k in ['plat', 'plate']) for d in vehicle_results):
            ocr_results = await OCRProcessor.process_plates(image, vehicle_results)

        if not ocr_results or ocr_results[0].get('status') != 'success':
            raise HTTPException(status_code=404, detail="Plat nomor tidak dapat dibaca dari gambar.")
        
        nomor_plat_terdeteksi = ocr_results[0]['text'].strip().upper()

        cursor = conn.cursor(dictionary=True)
        waktu_keluar_aktual = datetime.now()

        try:
            cursor.execute(
                "SELECT id_masuk, jenis_kendaraan, tipe_plat FROM kendaraan_masuk WHERE nomor_plat = %s AND status = 'Aktif' ORDER BY tanggal_masuk DESC LIMIT 1",
                (nomor_plat_terdeteksi,)
            )
            kendaraan_aktif = cursor.fetchone()
            if not kendaraan_aktif:
                raise HTTPException(status_code=404, detail=f"Tidak ditemukan kendaraan aktif dengan plat: {nomor_plat_terdeteksi}")
            
            id_masuk_kendaraan = kendaraan_aktif['id_masuk']

            info_parkir = hitung_biaya_parkir(conn, id_masuk_kendaraan, waktu_keluar_aktual)
            if not info_parkir:
                raise HTTPException(status_code=500, detail='Gagal menghitung biaya parkir.')

            cursor.execute(
                "INSERT INTO kendaraan_keluar (id_masuk, nomor_plat, tanggal_keluar) VALUES (%s, %s, %s)",
                (id_masuk_kendaraan, nomor_plat_terdeteksi, waktu_keluar_aktual)
            )
            id_keluar_baru = cursor.lastrowid

            cursor.execute(
                "INSERT INTO transaksi (id_masuk, id_keluar, durasi, biaya, status_pembayaran) VALUES (%s, %s, %s, %s, 'lunas')",
                (id_masuk_kendaraan, id_keluar_baru, info_parkir['durasi_menit'], info_parkir['biaya_rp'])
            )

            cursor.execute("UPDATE kendaraan_masuk SET status = 'Selesai' WHERE id_masuk = %s", (id_masuk_kendaraan,))
            
            conn.commit()

            detail_respons = {
                "nomor_plat": nomor_plat_terdeteksi, "jenis_kendaraan": kendaraan_aktif['jenis_kendaraan'],
                "tipe_plat": kendaraan_aktif['tipe_plat'], **info_parkir,
                "tanggal_keluar": waktu_keluar_aktual.isoformat()
            }
            return {
                "status": "success",
                "message": f"Proses keluar untuk plat '{nomor_plat_terdeteksi}' berhasil.",
                "annotated_image": ImageProcessor.to_base64(AnnotationProcessor.annotate_combined(image, vehicle_detections, sv.Detections.empty(), ocr_results)),
                "exit_details": {
                    "success": True, "message": f"Kendaraan {nomor_plat_terdeteksi} berhasil keluar.",
                    "detail_parkir": detail_respons
                }
            }

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            if cursor: cursor.close()

    except Exception as e:
        raise e
    
@app.get('/api/akses')
def get_semua_akses(conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, nomor_plat, status FROM akses ORDER BY nomor_plat ASC")
        return cursor.fetchall()
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        cursor.close()

@app.post('/api/akses', status_code=201)
def tambah_akses(item: AksesItem, conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO akses (nomor_plat, status) VALUES (%s, %s)", (item.nomor_plat.upper(), item.status))
        conn.commit()
        return {'success': True, 'message': f'Plat {item.nomor_plat.upper()} berhasil ditambahkan ke {item.status}.'}
    except Error as e:
        conn.rollback()
        if e.errno == 1062:
            raise HTTPException(status_code=409, detail=f'Plat nomor {item.nomor_plat.upper()} sudah ada di daftar.')
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        cursor.close()

@app.delete('/api/akses/{id_akses}')
def hapus_akses(id_akses: int, conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM akses WHERE id = %s", (id_akses,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail='ID tidak ditemukan di daftar akses.')
        return {'success': True, 'message': 'Plat berhasil dihapus dari daftar akses.'}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        cursor.close()

@app.get('/api/pengaturan')
def get_pengaturan(conn: mysql.connector.MySQLConnection = Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT setting_name, setting_value FROM pengaturan_sistem")
        return {item['setting_name']: item['setting_value'] for item in cursor.fetchall()}
    except Error as e:
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

@app.post('/api/pengaturan')
def save_pengaturan(
    settings_data: Dict[str, Any],
    conn: mysql.connector.MySQLConnection = Depends(get_db_connection)
):
    if not settings_data:
        raise HTTPException(status_code=400, detail='Tidak ada data yang dikirim')
    
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO pengaturan_sistem (setting_name, setting_value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)"
        data_to_save = list(settings_data.items())
        cursor.executemany(sql, data_to_save)
        conn.commit()
        return {'success': True, 'message': 'Pengaturan berhasil disimpan.'}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {str(e)}')
    finally:
        if cursor: cursor.close()

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
