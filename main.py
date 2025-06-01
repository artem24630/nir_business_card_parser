"""
Automated Business Card Data Extraction Pipeline
------------------------------------------------
Key capabilities
================
1. **Telegram Scraper** – downloads candidate business‑card images from a list of Telegram channels.
2. **Business‑Card Classifier** – filters the stream so that only true cards continue to OCR (two back‑ends: *heuristic* or *CNN*).
3. **OCR Layer** – configurable OCR back‑ends (Tesseract, EasyOCR, PaddleOCR) with automatic language detection.
4. **NER Layer** – spaCy‑based Named‑Entity‑Recognition tuned for business‑card fields.
5. **Contact Parser** – post‑processing that converts raw entities into a structured record.
6. **SQLite Store** – persists recognised contacts for downstream analysis.
7. **Experiment Harness** – utilities for benchmarking OCR & NER combinations on a labelled dev‑set.
~~~~~~~~~~~~~~~~~
"""
from __future__ import annotations

import os
import re
import io
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import en_core_web_trf
from cv2.dnn import Model

# Torch (optional – only needed for CNN classifier)
try:
    import torch
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# OCR back‑ends
import pytesseract  # Tesseract wrapper

try:
    import easyocr  # type: ignore

    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

# NER
import spacy
from spacy.tokens import Doc
from torchvision import models, transforms
import torch.nn as nn
# Telegram
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto

# -----------------------------------------------------------------------------
# 0.  Config & Globals
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s – %(message)s")

API_ID = 25613853
API_HASH = "70959962d3c518178e32dd7900cb79b0"
SESSION_NAME = os.getenv("TG_SESSION_NAME", "session")

DB_PATH = Path("business_cards.sqlite")
IMG_DIR = Path("data/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load spaCy NER (replace with your fine‑tuned model path if available)
try:
    NER_NLP = en_core_web_trf.load()  # spacy.load("en-core-web-trf")
except OSError:
    NER_NLP = spacy.load("ru_core_news_sm")

# Entity mapping from spaCy label -> canonical field
ENTITY_MAP = {
    "ORG": "company",
    "PERSON": "contact_name",
    "EMAIL": "email",
    "PHONE": "phone",
    "URL": "website",
    "GPE": "address",  # fallback
}


# -----------------------------------------------------------------------------
# 1.  Telegram Scraper
# -----------------------------------------------------------------------------

def download_business_card_images(channel_usernames: List[str], limit_per_channel: int = 200) -> List[Path]:
    """Connects via Telethon and downloads images from the supplied channels."""
    if not API_ID or not API_HASH:
        raise RuntimeError("Set TG_API_ID and TG_API_HASH env variables first!")

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    images: List[Path] = []

    async def _scrape():
        await client.start()
        for ch in channel_usernames:
            logging.info(f"Scanning {ch}")
            async for msg in client.iter_messages(ch, limit=limit_per_channel):
                if isinstance(msg.media, MessageMediaPhoto):
                    fn = IMG_DIR / f"{msg.id}_{ch.strip('@')}.jpg"
                    if not fn.exists():
                        await msg.download_media(file=fn)
                    images.append(fn)
        await client.disconnect()

    client.loop.run_until_complete(_scrape())
    logging.info(f"Downloaded {len(images)} images")
    return images


# -----------------------------------------------------------------------------
# 2.  Business‑Card Image Classifier
# -----------------------------------------------------------------------------

# 2.1  Heuristic detector ------------------------------------------------------

def _heuristic_card_detector(image_path: Path) -> bool:
    """Fast OpenCV‑based heuristic: looks for a dominant rectangular contour with
    business‑card‑like aspect ratio and sufficient text density."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # Find the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    # A business card is roughly rectangular (4 vertices) and occupies a large
    # fraction of the image.
    h, w = gray.shape
    area = cv2.contourArea(largest)
    img_area = h * w
    rectangular = len(approx) == 4 and area > 0.1 * img_area

    if not rectangular:
        return False

    # Aspect ratio check (width > height; typical card 1.4–1.8 ratio)
    x, y, cw, ch = cv2.boundingRect(approx)
    ratio = cw / ch if ch else 0
    if ratio < 1.2 or ratio > 2.5:
        return False

    # Text density: run a quick adaptive threshold & count dark pixels
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    text_pixels = np.count_nonzero(thresh)
    density = text_pixels / img_area
    return density > 0.05  # at least 5% of pixels look like text


# 2.2  CNN detector ------------------------------------------------------------

_CNN_MODEL: Optional[torch.nn.Module] = None


def _load_cnn_model(model_path: Path = Path("card_classifier.pt")) -> Optional[torch.nn.Module]:
    if not _TORCH_AVAILABLE or not model_path.exists():
        return None
        # 1) Создадим модель ResNet-18 и настроим head
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 класса: card/not_card

    # 2) Загружаем веса (state_dict)
    state = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


_CNN_MODEL = _load_cnn_model()
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) if _TORCH_AVAILABLE else None


def _cnn_card_detector(image_path: Path, threshold: float = 0.5) -> bool:
    if _CNN_MODEL is None or _TRANSFORM is None:
        return False  # fallback handled by caller
    img = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = _CNN_MODEL(tensor)  # shape (1,2)
        prob_card = torch.softmax(logits, dim=1)[0, 0].item()
        print(prob_card)
        return prob_card >= threshold


# 2.3  Public API --------------------------------------------------------------

def is_business_card(image_path: Path, method: str = "auto") -> bool:
    """Return True if the image *looks* like a business card.

    Parameters
    ----------
    image_path: Path to image.
    method: "auto" – use CNN if available otherwise heuristic.
            "cnn"  – force CNN (raises if model unavailable).
            "heuristic" – force OpenCV heuristic.
    """
    method = method.lower()
    if method == "heuristic":
        return _heuristic_card_detector(image_path)
    elif method == "cnn":
        if _CNN_MODEL is None:
            raise RuntimeError("CNN model not loaded – place card_classifier.pt next to the script")
        return _cnn_card_detector(image_path)
    elif method == "auto":
        if _CNN_MODEL is not None:
            return _cnn_card_detector(image_path) or _heuristic_card_detector(image_path)
        return _heuristic_card_detector(image_path)
    else:
        raise ValueError(f"Unknown classifier method: {method}")


# -----------------------------------------------------------------------------
# 3.  OCR Back‑ends
# -----------------------------------------------------------------------------

def ocr_tesseract(img: Image.Image, lang: str = "rus+eng") -> str:
    return pytesseract.image_to_string(img, lang=lang, config="--psm 6")


def ocr_easyocr(img: Image.Image, lang: str = "en") -> str:
    if not _EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr is not installed – `pip install easyocr`. ")
    reader = easyocr.Reader([lang])
    res = reader.readtext(np.array(img))
    return "\n".join([t for _, t, _ in res])


def perform_ocr(image_path: Path, method: str = "easyocr") -> str:
    img = Image.open(image_path)
    if method == "tesseract":
        return ocr_tesseract(img)
    elif method == "easyocr":
        return ocr_easyocr(img)
    else:
        raise ValueError(f"Unknown OCR method: {method}")


# -----------------------------------------------------------------------------
# 4.  NER & Post‑processing
# -----------------------------------------------------------------------------
def guess_company(text_lines):
    text_lines = [line.strip() for line in text_lines.split("\n") if line.strip()]
    print(text_lines)
    company_keywords = [
        "hotel", "resort", "group", "llc", "ltd", "corporation", "company", "inc", "s.a.", "s.r.o.", "gmbh", "co.",
        "spa", "holdings", "services", "hospitality", "travel"
    ]
    candidates = []

    # Пройтись по первым 5 строкам
    for line in text_lines:
        line_clean = line.strip().lower()
        if not line_clean:
            continue

        # Если есть ключевое слово — приоритет
        if any(keyword in line_clean for keyword in company_keywords):
            candidates.append((line, 2))
        # Если полностью в UPPER CASE — вероятно, имя компании
        elif line.isupper() and len(line) < 50:
            candidates.append((line, 1))
        # Первая непустая строка — слабый сигнал
        elif len(candidates) == 0:
            candidates.append((line, 0.5))

    # Вернуть строку с максимальным весом
    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return [candidates[0][0]]

    return []


def extract_entities(text: str) -> Dict[str, List[str]]:
    doc: Doc = NER_NLP(text)
    result: Dict[str, List[str]] = {}
    for ent in doc.ents:
        key = ENTITY_MAP.get(ent.label_, ent.label_)
        result.setdefault(key, []).append(ent.text)

    if not result.get("company"):
        result.setdefault('company', []).extend(guess_company(text))

    # Regex fallbacks for phone/email/url if NER missed them
    email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    phone_re = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
    url_re = re.compile(r"https?://[\w./-]+|www\.[\w./-]+")

    for pattern, field in [(email_re, "email"), (phone_re, "phone"), (url_re, "website")]:
        for m in pattern.findall(text):
            result.setdefault(field, []).append(m)

    return result


# -----------------------------------------------------------------------------
# 5.  Contact Info Extractor  (Integrate your notebook code here)
# -----------------------------------------------------------------------------

def extract_contact_info_from_image(image_path: Path, ocr_method: str = "easyocr") -> Dict[str, List[str]]:
    raw_text = perform_ocr(image_path, method=ocr_method)
    entities = extract_entities(raw_text)
    return entities


# -----------------------------------------------------------------------------
# 6.  SQLite Storage
# -----------------------------------------------------------------------------

def init_db(path: Path = DB_PATH):
    """Create table 'cards' with one row per business card."""
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        # Enable JSON1 extension if not default
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.execute(
            """CREATE TABLE IF NOT EXISTS cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_file TEXT UNIQUE,
                    phone TEXT,
                    company TEXT,
                    email TEXT,
                    address TEXT,
                    other_fields TEXT,
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
        )
        conn.commit()


def save_contact_record(image_file: Path, data: Dict[str, List[str]], path: Path = DB_PATH):
    """Insert or replace a row in 'cards'."""
    # Extract primary fields (taking first value if multiple)
    phone_vals = data.get("phone", [])
    company_vals = data.get("company", [])
    email_vals = data.get("email", [])
    address_vals = data.get("address", [])

    phone = "; ".join(phone_vals) if phone_vals else None
    company = "; ".join(company_vals) if company_vals else None
    email = "; ".join(email_vals) if email_vals else None
    address = "; ".join(address_vals) if address_vals else None

    # Build JSON for other fields
    other = {}
    for key, vals in data.items():
        if key not in ("phone", "company", "email", "address"):
            other[key] = vals
    other_json = json.dumps(other, ensure_ascii=False) if other else None

    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO cards
               (image_file, phone, company, email, address, other_fields)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(image_file),
                phone,
                company,
                email,
                address,
                other_json,
            ),
        )
        conn.commit()


# -----------------------------------------------------------------------------
# 7.  Experiment Utilities
# -----------------------------------------------------------------------------
from rapidfuzz import fuzz


def fuzzy_match(pred_set, truth_set, threshold=70):
    tp = 0
    used_truth = set()
    for pred in pred_set:
        for truth in truth_set:
            if truth in used_truth:
                continue
            if fuzz.ratio(pred.lower(), truth.lower()) >= threshold:
                tp += 1
                used_truth.add(truth)
                break
    fp = len(pred_set) - tp
    fn = len(truth_set) - tp
    return tp, fp, fn


def evaluate(dev_images, threshold=90):
    from collections import Counter
    tp_total, fp_total, fn_total = Counter(), Counter(), Counter()
    for img, truth in dev_images:
        pred = extract_contact_info_from_image(img, ocr_method="easyocr")
        all_keys = set(truth) | set(pred)
        for field in all_keys:
            truth_set = set(truth.get(field, []))
            pred_set = set(pred.get(field, []))
            tp, fp, fn = fuzzy_match(pred_set, truth_set, threshold)
            tp_total[field] += tp
            fp_total[field] += fp
            fn_total[field] += fn
    print(f"\n==== OCR = easyocr ====")
    for field in tp_total:
        precision = tp_total[field] / (tp_total[field] + fp_total[field]) if (tp_total[field] + fp_total[field]) else 0
        recall = tp_total[field] / (tp_total[field] + fn_total[field]) if (tp_total[field] + fn_total[field]) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"{field:12s} P={precision:.2f} R={recall:.2f} F1={f1:.2f}")


# def evaluate(dev_images: List[Tuple[Path, Dict[str, List[str]]]]) -> None:
#     from collections import Counter
#     tp, fp, fn = Counter(), Counter(), Counter()
#     for img, truth in dev_images:
#         pred = extract_contact_info_from_image(img, ocr_method="easyocr")
#         for field in set(truth) | set(pred):
#             truth_set = set(truth.get(field, []))
#             pred_set = set(pred.get(field, []))
#             tp[field] += len(truth_set & pred_set)
#             fp[field] += len(pred_set - truth_set)
#             fn[field] += len(truth_set - pred_set)
#     logging.info("\n==== OCR=%s ====" % "easyocr")
#     for field in tp:
#         precision = tp[field] / (tp[field] + fp[field]) if (tp[field] + fp[field]) else 0
#         recall = tp[field] / (tp[field] + fn[field]) if (tp[field] + fn[field]) else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
#         logging.info(f"{field:12s} P={precision:.2f} R={recall:.2f} F1={f1:.2f}")


# -----------------------------------------------------------------------------
# 8.  End‑to‑end Pipeline Driver
# -----------------------------------------------------------------------------

def run_full_pipeline(channels: List[str], limit: int = 200,
                      ocr_method: str = "easyocr", cls_method: str = "auto"):
    init_db()
    images = download_business_card_images(channels, limit_per_channel=limit)
    for img_path in images:
        if not is_business_card(img_path, method=cls_method):
            logging.info(f"Skipped {img_path.name} – not a card")
            continue
        data = extract_contact_info_from_image(img_path, ocr_method=ocr_method)
        save_contact_record(img_path, data)
        logging.info(f"Processed {img_path.name}: {sum(len(v) for v in data.values())} fields")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    CHANNELS = ["@business_card_hotels"]
    run_full_pipeline(CHANNELS, limit=500, ocr_method="easyocr", cls_method="auto")
    # dev_images = [
    #     (
    #         Path("data/images/2045_business_card_hotels.jpg"),
    #         {
    #             "company": ["Prestige Garden Hotel"],
    #             "phone": ["0090 (252) 417 59 75", "0090 533 190 63 48"],
    #             "email": ["reservation@prestigegardenhotel.com"],
    #             "website": ["www.prestigegardenhotel.com"],
    #             "address": ["Siteler-Marmaris-Türkiye"]
    #         }
    #     ),
    #     (
    #         Path("data/images/2046_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Gizem Yıldız Özgözen"],
    #             "position": ["Asst. Public Relations Manager"],
    #             "company": ["AQI Pegasos World"],
    #             "address": ["Titreyengöl Mevkii, Side, ANTALYA, TÜRKİYE"],
    #             "phone": ["+90 242 756 98 00", "+90 242 756 97 98", "+90 507 824 69 21"],
    #             "email": ["gizem.yildiz@tuihotels.com.tr"],
    #             "website": ["www.tthotels.com"]
    #         }
    #     ),
    #     (
    #         Path("data/images/2038_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Daria Grudinina"],
    #             "position": ["Sales Executive"],
    #             "company": ["Rixos Hotels"],
    #             "address": ["Meltem Mahallesi, Sakıp Sabancı Blv. No:3, 07030 Konyaaltı / Antalya / Türkiye"],
    #             "phone": ["+90 242 249 49 49", "+90 501 510 11 11"],
    #             "fax": ["+90 242 249 49 00"],
    #             "email": ["daria.grudinina@rixos.com"],
    #             "website": ["rixos.com"]
    #         }
    #     ),
    #     (
    #         Path("data/images/1436_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Michael Magdy"],
    #             "position": ["Assistant Director of Sales & E-Distribution"],
    #             "company": ["Savoy Group"],
    #             "email": ["michael.magdy@savoy-sharm.com"],
    #             "phone": ["(+2-02) 227 35 710", "(+2-02) 227 35 780", "(+2) 0122 747 80 12", "(+44) 1844 318 291"],
    #             "fax": ["(+2-02) 227 35 495"],
    #             "hotline": ["(+20-2) 19846"],
    #             "address": [
    #                 "P.O Box 169, SOHO Square, Sharm El Sheikh, South Sinai, Egypt",
    #                 "Villa 26, Atteya El Sawalhi St., District 8, Nasr City, Cairo, Egypt"
    #             ],
    #             "website": ["www.savoygroup-sharm.com"]
    #         }
    #     ),
    #     (
    #         Path("data/images/1437_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Mina Botros"],
    #             "position": ["Director of Sales"],
    #             "company": ["Gravity Hotels & Resorts"],
    #             "phone": ["(+20) 1010 49 8827"],
    #             "email": ["mina@hotelsgravity.com"],
    #             "website": ["www.hotelsgravity.com"]
    #         }
    #     ),
    #     (
    #         Path("data/images/1440_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Nil Burkay"],
    #             "position": ["Sales & Marketing Asst. Manager"],
    #             "company": ["Amon Hotels"],
    #             "address": ["Kadriye Mh. Deniz Cd. No:119/1, Belek / Serik / Antalya"],
    #             "phone": ["+90 242 727 26 60", "+90 546 282 44 11"],
    #             "fax": ["+90 242 727 26 69"],
    #             "email": ["nilburkay@amonhotels.com"],
    #             "website": ["amonhotels.com"]
    #         }
    #     ),
    #     (
    #         Path("data/images/1446_business_card_hotels.jpg"),
    #         {
    #             "company": ["Sunset Beach Resort & Spa"],
    #             "address": ["100C/2 Tran Hung Dao, Duong Dong, Phu Quoc, Kien Giang"],
    #             "phone": ["(+84) 2973 567 888"],
    #             "email": ["info@sunset.vn"],
    #             "website": ["sunsetbeach.vn"],
    #             "facebook": ["Sunset Beach Resort and Spa"]
    #         }
    #     ),
    #     (
    #         Path("data/images/2031_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Ainura Artykbekova"],
    #             "position": ["Sales Executive"],
    #             "company": ["Mandarin Oriental, Bodrum"],
    #             "address": ["Gölköy Mahallesi, 314. Sokak No. 10, 48400, Bodrum, Muğla, Turkey"],
    #             "phone": ["+90 (0) 252 311 1888", "+90 (0) 549 802 0954"],
    #             "email": ["aartybekova@mohg.com"],
    #             "website": ["mandarinoriental.com"]
    #         }
    #     ),
    #     (
    #         Path("data/images/1450_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Thanat Wongin"],
    #             "position": ["Director of Sales"],
    #             "company": ["Sugar Marina Hotel"],
    #             "phone": ["66 (0) 83 451 4654"],
    #             "email": ["dos@sugarmarinahotel.com"],
    #             "website": ["sugarmarinahotel.com"],
    #             "facebook": ["sugarmarinahotel"]
    #         }
    #     ),
    #     (
    #         Path("data/images/1473_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Samiksha Chudal"],
    #             "position": ["Sales Manager – Groups & Events"],
    #             "company": ["Centara Mirage Beach Resort Dubai"],
    #             "email": ["samikshach@chr.co.th"],
    #             "phone": ["+971 55 811 7425", "+971 4 522 9999"],
    #             "address": ["P.O. Box 31308, Dubai Islands, Dubai, United Arab Emirates"],
    #             "facebook": ["CentaraMirageDubai"]
    #         }
    #     ),
    #     (
    #         Path("data/images/IMG_20200209_183310.jpg"),
    #         {
    #             "contact_name": ["CA Pushkar S. Solanki"],
    #             "position": ["B.Com, ACA"],
    #             "address": ["Flat No. 04, Dwarka Complex, Savedi, Near Hotel Premdan, Ahmednagar 414 003"],
    #             "email": ["capushkarsolanki@gmail.com"],
    #             "phone": ["+91 9730704929", "+91 9421833600"]
    #         }
    #     ),
    #     (
    #         Path("data/images/1432_business_card_hotels.jpg"),
    #         {
    #             "contact_name": ["Shaza Ragaey"],
    #             "position": ["Director of Sales & Marketing"],
    #             "company": ["Amphoras Hotels"],
    #             "email": ["Shaza.ragaey@amphorashotels.com"],
    #             "address": [
    #                 "Sharm El Sheikh, Om El Seid Cliff, 110 South Sinai - Egypt",
    #                 "Egypt Post Code: 12411",
    #                 "Sharm El Sheikh Post Code: 45625"
    #             ],
    #             "phone": ["(+2) 069 366 1815", "(+2) 01222930569"],
    #             "fax": ["(+2) 069 366 1697"],
    #             "website": ["www.amphorashotels.com"]
    #         }
    #     )
    # ]
    # evaluate(dev_images)
