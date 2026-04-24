import io
import json
import os
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import fitz
import numpy as np
import torch
import torchvision.transforms.functional as TF
from flask import Blueprint, Response, jsonify, request
from PIL import Image, ImageFilter, ImageOps
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

from helpers import text_to_pdf_buffer, upload_pdf_to_s3, upload_vectorstore_to_s3
from ocr import refine_handwritten_notes
from utils import process_pdf_rag


AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY_PATH = TRAINED_MODELS_DIR / "registry.json"

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION,
)

finetune = Blueprint("finetune", __name__)
progress_queues: Dict[str, "queue.Queue[str]"] = {}
model_registry_lock = threading.Lock()
loaded_model_cache: Dict[str, Tuple[TrOCRProcessor, VisionEncoderDecoderModel]] = {}

device = "cpu"
BASE_MODEL_NAME = "microsoft/trocr-base-handwritten"


def read_model_registry():
    with model_registry_lock:
        if not MODEL_REGISTRY_PATH.exists():
            return []
        try:
            return json.loads(MODEL_REGISTRY_PATH.read_text())
        except Exception:
            return []


def write_model_registry(records):
    with model_registry_lock:
        MODEL_REGISTRY_PATH.write_text(json.dumps(records, indent=2))


def upsert_model_record(record):
    records = read_model_registry()
    updated = False
    for index, item in enumerate(records):
        if item.get("id") == record["id"]:
            records[index] = {**item, **record}
            updated = True
            break
    if not updated:
        records.append(record)
    write_model_registry(records)
    return record


def update_model_record(model_id, **updates):
    records = read_model_registry()
    for index, item in enumerate(records):
        if item.get("id") == model_id:
            records[index] = {**item, **updates}
            write_model_registry(records)
            return records[index]
    return None


def resolve_huggingface_snapshot(model_name):
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path), True

    repo_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name, False

    snapshot_candidates = sorted(
        (
            path
            for path in snapshots_dir.iterdir()
            if path.is_dir()
            and (path / "config.json").exists()
            and (path / "preprocessor_config.json").exists()
            and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
            and (((path / "vocab.json").exists() and (path / "merges.txt").exists()) or (path / "sentencepiece.bpe.model").exists())
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if snapshot_candidates:
        return str(snapshot_candidates[0]), True

    return model_name, False


def load_base_model_components():
    model_source, local_files_only = resolve_huggingface_snapshot(BASE_MODEL_NAME)
    if not local_files_only:
        raise RuntimeError(
            f"The base handwriting model '{BASE_MODEL_NAME}' is not available in the local Hugging Face cache."
        )

    try:
        processor = TrOCRProcessor.from_pretrained(model_source, local_files_only=True)
        model = VisionEncoderDecoderModel.from_pretrained(model_source, local_files_only=True).to(device)
        return processor, model
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the local base handwriting model from '{model_source}'. {exc}"
        ) from exc


def get_model_record(model_id):
    for record in read_model_registry():
        if record.get("id") == model_id:
            return record
    return None


def delete_model_record(model_id):
    records = read_model_registry()
    remaining = [record for record in records if record.get("id") != model_id]
    write_model_registry(remaining)


def build_augment_pipeline(img_size=384):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply([transforms.RandomRotation(degrees=2)], p=0.15),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.02, 0.02),
                        scale=(0.98, 1.02),
                        shear=1.5,
                    )
                ],
                p=0.2,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.Lambda(lambda image: TF.adjust_sharpness(image, sharpness_factor=1.1)),
            transforms.Lambda(lambda image: image.convert("RGB")),
        ]
    )


class TrOCRJsonDataset(Dataset):
    def __init__(self, items, processor, bucket=None, augment=True):
        self.items = items
        self.processor = processor
        self.bucket = bucket
        self.augment = augment
        self.aug = build_augment_pipeline() if augment else None

    def __len__(self):
        return len(self.items)

    def _load_image(self, key_or_path: str):
        if os.path.exists(key_or_path):
            return Image.open(key_or_path).convert("RGB")
        buffer = io.BytesIO()
        s3.download_fileobj(self.bucket, key_or_path, buffer)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __getitem__(self, idx):
        item = self.items[idx]
        image = self._load_image(item["s3ImageKey"])
        if self.aug:
            image = self.aug(image)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        tokenized = self.processor.tokenizer(
            item["label"],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels = torch.where(labels == pad_token_id, torch.tensor(-100), labels)
        return {"pixel_values": pixel_values, "labels": labels}


def normalize_model_name(model_name, sample_count):
    cleaned = (model_name or "").strip()
    if cleaned:
        return cleaned
    return f"My Handwriting Model {sample_count} samples"


def preprocess_handwriting_page(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = image.filter(ImageFilter.SHARPEN)
    return image


def find_line_regions(image):
    image_array = np.array(image)
    threshold = max(125, int(np.percentile(image_array, 35)))
    ink_mask = image_array < threshold
    row_density = ink_mask.mean(axis=1)
    min_row_density = max(0.005, float(row_density.mean() * 1.25))
    active_rows = np.where(row_density > min_row_density)[0]

    if len(active_rows) == 0:
        return [(0, 0, image.width, image.height)]

    line_regions = []
    start = active_rows[0]
    previous = active_rows[0]
    gap_threshold = max(14, image.height // 170)

    for row in active_rows[1:]:
        if row - previous > gap_threshold:
            line_regions.append((start, previous))
            start = row
        previous = row
    line_regions.append((start, previous))

    boxes = []
    for top, bottom in line_regions:
        if bottom - top < 12:
            continue

        vertical_padding = max(12, image.height // 260)
        crop_mask = ink_mask[max(top - vertical_padding, 0):min(bottom + vertical_padding, image.height), :]
        col_density = crop_mask.mean(axis=0)
        min_col_density = max(0.0025, float(col_density.mean() * 1.15))
        active_cols = np.where(col_density > min_col_density)[0]
        if len(active_cols) == 0:
            continue

        horizontal_padding = max(20, image.width // 180)
        left = max(int(active_cols[0]) - horizontal_padding, 0)
        right = min(int(active_cols[-1]) + horizontal_padding, image.width)
        upper = max(int(top) - vertical_padding, 0)
        lower = min(int(bottom) + vertical_padding, image.height)
        boxes.append((left, upper, right, lower))

    return boxes or [(0, 0, image.width, image.height)]


def get_custom_model_components(model_id):
    cached = loaded_model_cache.get(model_id)
    if cached is not None:
        return cached

    record = get_model_record(model_id)
    if not record or record.get("status") != "ready":
        raise RuntimeError("Selected handwriting model is not ready.")

    model_dir = record.get("model_dir")
    if not model_dir or not Path(model_dir).exists():
        raise RuntimeError("Trained model files are missing on disk.")

    processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    loaded_model_cache[model_id] = (processor, model)
    return processor, model


def decode_line_with_model(processor, model, image):
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def run_custom_model_ocr(bucket, key, model_id):
    processor, model = get_custom_model_components(model_id)
    obj = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = obj["Body"].read()

    result_parts = []
    ocr_pages = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    for index in range(len(pdf)):
        page = pdf[index]
        pix = page.get_pixmap(dpi=400)
        processed_image = preprocess_handwriting_page(pix.tobytes("png"))
        extracted_lines = []

        for box in find_line_regions(processed_image):
            line_image = processed_image.crop(box)
            if line_image.width < 80 or line_image.height < 24:
                continue
            line_text = decode_line_with_model(processor, model, line_image)
            if line_text:
                extracted_lines.append(line_text)

        page_text = "\n".join(extracted_lines).strip()
        if page_text:
            result_parts.append(page_text)
            ocr_pages.append({"page": index + 1, "text": page_text})

    final_text = "\n\n".join(result_parts).strip()
    if not final_text:
        raise RuntimeError("No handwritten text could be extracted with the selected model.")

    refined_text = refine_handwritten_notes(final_text)
    pdf_bytes = text_to_pdf_buffer(refined_text)
    pdf_url = upload_pdf_to_s3(pdf_bytes, bucket)

    upload_dir = BASE_DIR / "uploadfiles"
    vectorstore_dir = BASE_DIR / "vectorstores"
    upload_dir.mkdir(parents=True, exist_ok=True)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    tmp_pdf_path = upload_dir / f"{uuid.uuid4()}.pdf"
    tmp_pdf_path.write_bytes(pdf_bytes)

    persist_dir = vectorstore_dir / str(uuid.uuid4())
    process_pdf_rag(str(tmp_pdf_path), str(persist_dir))
    upload_vectorstore_to_s3(str(persist_dir), bucket)

    if persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)
    if tmp_pdf_path.exists():
        tmp_pdf_path.unlink()

    return {
        "pdf_url": pdf_url,
        "vectorstore": str(persist_dir),
        "ocr_pages": ocr_pages,
        "raw_text": final_text,
        "refined_text": refined_text,
        "model_id": model_id,
    }


def train_job(job_id: str, dataset_list: List[Dict], bucket: str, user_id: str, model_name: str):
    progress_queue = progress_queues[job_id]

    def emit(message_type, payload):
        progress_queue.put(json.dumps({"type": message_type, **payload}))

    save_dir = TRAINED_MODELS_DIR / job_id

    try:
        emit("status", {"message": "Using CPU fast fine-tune preset"})

        processor, model = load_base_model_components()

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
        model.config.decoder.vocab_size = processor.tokenizer.vocab_size

        dataset = TrOCRJsonDataset(dataset_list, processor, bucket=bucket, augment=True)
        batch_size = 2 if len(dataset) >= 2 else 1
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        emit("status", {"message": "Phase 1: Fast decoder tuning"})
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

        optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5)
        steps_total = max(1, len(loader))
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_total)

        step = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            emit(
                "progress",
                {
                    "phase": 1,
                    "epoch": 1,
                    "percent": int(step * 100 / steps_total),
                    "loss": round(loss.item(), 4),
                },
            )

        emit("status", {"message": "Phase 2: Quick full-model tune"})
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True

        optimizer = AdamW(model.parameters(), lr=2e-6)
        steps_total = max(1, len(loader))
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_total)
        best_loss = float("inf")
        save_dir.mkdir(parents=True, exist_ok=True)
        step = 0

        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            emit(
                "progress",
                {
                    "phase": 2,
                    "epoch": 1,
                    "percent": int(step * 100 / steps_total),
                    "loss": round(loss.item(), 4),
                },
            )
            if loss.item() < best_loss:
                best_loss = loss.item()
                model.save_pretrained(save_dir, max_shard_size="500MB")
                processor.save_pretrained(save_dir)

        record = update_model_record(
            job_id,
            status="ready",
            model_dir=str(save_dir),
            updated_at=datetime.now(timezone.utc).isoformat(),
            loss=round(best_loss, 4),
        )

        emit("done", {"ok": True, "model_id": job_id, "model_name": model_name, "model_dir": str(save_dir), "record": record})
    except Exception as exc:
        update_model_record(
            job_id,
            status="failed",
            last_error=str(exc),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        emit("error", {"ok": False, "error": str(exc), "trace": traceback.format_exc()})
    finally:
        progress_queue.put(json.dumps({"type": "finished"}))


@finetune.route("/start_finetune", methods=["POST"])
def start_finetune():
    body = request.get_json(force=True)
    dataset = body.get("dataset", [])
    bucket = body.get("bucket") or os.getenv("AWS_S3_BUCKET")
    user_id = body.get("userID") or "demo"
    model_name = normalize_model_name(body.get("model_name"), len(dataset))

    if len(dataset) < 3:
        return jsonify({"error": "Upload at least 3 handwriting samples before training."}), 400

    job_id = str(int(time.time() * 1000))
    record = {
        "id": job_id,
        "name": model_name,
        "user_id": user_id,
        "status": "training",
        "sample_count": len(dataset),
        "base_model": BASE_MODEL_NAME,
        "model_dir": "",
        "last_error": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    upsert_model_record(record)

    progress_queues[job_id] = queue.Queue()
    threading.Thread(target=train_job, args=(job_id, dataset, bucket, user_id, model_name), daemon=True).start()
    return jsonify({"job_id": job_id, "device": device, "model_name": model_name})


@finetune.route("/events/<job_id>")
def events(job_id):
    if job_id in progress_queues:
        progress_queue = progress_queues[job_id]

        def stream():
            yield "retry: 1500\n\n"
            while True:
                try:
                    message = progress_queue.get(timeout=1)
                    yield f"data: {message}\n\n"
                    parsed = json.loads(message)
                    if parsed.get("type") in ("done", "error", "finished"):
                        break
                except queue.Empty:
                    yield ":\n\n"

        return Response(stream(), mimetype="text/event-stream")
    return Response(iter(["retry: 1500\n\n", 'data: {"type":"finished"}\n\n']), mimetype="text/event-stream")


@finetune.route("/result/<job_id>")
def result(job_id):
    record = get_model_record(job_id)
    return jsonify({"ready": record is not None and record.get("status") == "ready", "model": record})


@finetune.route("/models", methods=["GET"])
def list_models():
    user_id = request.args.get("userID")
    records = read_model_registry()
    if user_id:
        records = [record for record in records if record.get("user_id") == user_id]
    records.sort(key=lambda record: record.get("created_at", ""), reverse=True)
    return jsonify({"models": records})


@finetune.route("/models/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    record = get_model_record(model_id)
    if record is None:
        return jsonify({"error": "Model not found."}), 404

    model_dir = record.get("model_dir")
    if model_dir:
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)

    loaded_model_cache.pop(model_id, None)
    delete_model_record(model_id)
    return jsonify({"ok": True, "deleted_id": model_id})


@finetune.route("/ocr_with_model", methods=["POST"])
def ocr_with_model():
    try:
        body = request.get_json(force=True)
        key = body.get("key") or ""
        bucket = body.get("bucket") or os.getenv("AWS_S3_BUCKET")
        model_id = body.get("model_id") or ""

        if not key or not bucket or not model_id:
            return jsonify({"message": "key, bucket, and model_id are required."}), 400

        payload = run_custom_model_ocr(bucket, key, model_id)
        return jsonify({"pdflink": payload}), 200
    except Exception as exc:
        return jsonify({"message": str(exc)}), 500
