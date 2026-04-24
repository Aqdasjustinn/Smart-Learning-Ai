import io
import os
import shutil
import uuid
from pathlib import Path

import boto3
import fitz
import pytesseract
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image

import env_config  # noqa: F401
from helpers import text_to_pdf_buffer, upload_pdf_to_s3, upload_vectorstore_to_s3
from openrouter_client import prompt_completion
from utils import process_pdf_rag


TESSERACT_CMD = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
HANDWRITING_OCR_PROVIDER = os.getenv("HANDWRITING_OCR_PROVIDER") or "google_vision"
GOOGLE_VISION_CREDENTIALS_PATH = os.getenv("GOOGLE_VISION_CREDENTIALS_PATH") or "../fourth-amp-476617-j8-f7a43bf7a0c0.json"

_google_vision_client = None

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="eu-north-1",
)


def extract_pdf(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = obj["Body"].read()

    result = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    for i in range(len(pdf)):
        page = pdf[i]
        extracted = page.get_text()

        if extracted.strip():
            result.append({i + 1: extracted})
        else:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img, lang="eng")
            result.append({i + 1: text})

    return result


def require_tesseract():
    if not pytesseract.pytesseract.tesseract_cmd or not Path(pytesseract.pytesseract.tesseract_cmd).exists():
        raise RuntimeError(
            "Tesseract is not available. Install tesseract or set TESSERACT_CMD in python-app/.env."
        )


def resolve_google_vision_credentials_path():
    configured_path = Path(GOOGLE_VISION_CREDENTIALS_PATH).expanduser()
    if configured_path.is_absolute() and configured_path.exists():
        return configured_path

    repo_relative = (Path(__file__).resolve().parent / configured_path).resolve()
    if repo_relative.exists():
        return repo_relative

    root_relative = (Path(__file__).resolve().parent.parent / configured_path).resolve()
    if root_relative.exists():
        return root_relative

    default_repo_path = Path(__file__).resolve().parent.parent / "fourth-amp-476617-j8-f7a43bf7a0c0.json"
    if default_repo_path.exists():
        return default_repo_path

    return None


def get_google_vision_client():
    global _google_vision_client

    if _google_vision_client is not None:
        return _google_vision_client

    credentials_path = resolve_google_vision_credentials_path()
    if credentials_path is None:
        return None

    credentials = service_account.Credentials.from_service_account_file(str(credentials_path))
    _google_vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    return _google_vision_client


def extract_handwritten_text_with_google_vision(image_bytes):
    client = get_google_vision_client()
    if client is None:
        raise RuntimeError(
            "Google Vision credentials were not found. Set GOOGLE_VISION_CREDENTIALS_PATH in python-app/.env."
        )

    response = client.document_text_detection(image=vision.Image(content=image_bytes))
    if response.error.message:
        raise RuntimeError(f"Google Vision OCR failed: {response.error.message}")

    annotation = response.full_text_annotation
    if annotation and annotation.text:
        return annotation.text.strip()

    text_annotation = response.text_annotations
    if text_annotation:
        return text_annotation[0].description.strip()

    return ""


def extract_handwritten_text(image_bytes):
    if HANDWRITING_OCR_PROVIDER != "google_vision":
        raise RuntimeError("Handwritten OCR is configured to use only Google Vision.")

    image = Image.open(io.BytesIO(image_bytes))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return extract_handwritten_text_with_google_vision(output.getvalue())


def refine_handwritten_notes(raw_text):
    cleaned_input = (raw_text or "").strip()
    if not cleaned_input:
        return ""

    prompt = f"""
You are cleaning OCR output from handwritten class notes.

Rules:
- Preserve the original meaning and order.
- Do not add new facts, examples, or explanations.
- Do not summarize or shorten the notes.
- Fix obvious OCR mistakes like broken words, punctuation, spacing, and line breaks.
- Keep technical terms when present.
- Turn fragmented lines into readable, organized notes.
- Use headings and bullet points only when they match the existing content.
- If a line is uncertain, keep it close to the OCR instead of inventing text.
- Return only the cleaned notes text.

Raw OCR text:
{cleaned_input}
"""

    refined_text = prompt_completion(prompt, temperature=0.1)
    return refined_text.strip() or cleaned_input


def run_vision(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        file_bytes = obj["Body"].read()

        result_parts = []
        ocr_pages = []
        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        for i in range(len(pdf)):
            page = pdf[i]
            pix = page.get_pixmap(dpi=400)
            img_bytes = pix.tobytes("png")
            ocr_text = extract_handwritten_text(img_bytes)
            if ocr_text:
                result_parts.append(ocr_text)
                ocr_pages.append({
                    "page": i + 1,
                    "text": ocr_text,
                })

        final_text = "\n\n".join(result_parts).strip()
        if not final_text:
            raise RuntimeError("No handwritten text could be extracted from the PDF.")

        refined_text = refine_handwritten_notes(final_text)
        pdf_bytes = text_to_pdf_buffer(refined_text)
        pdf_url = upload_pdf_to_s3(pdf_bytes, bucket)

        upload_dir = Path("./uploadfiles")
        vectorstore_dir = Path("./vectorstores")
        upload_dir.mkdir(parents=True, exist_ok=True)
        vectorstore_dir.mkdir(parents=True, exist_ok=True)

        tmp_pdf_path = upload_dir / f"{uuid.uuid4()}.pdf"
        with open(tmp_pdf_path, "wb") as file_obj:
            file_obj.write(pdf_bytes)

        persist_dir = vectorstore_dir / str(uuid.uuid4())
        process_pdf_rag(str(tmp_pdf_path), str(persist_dir))
        vectorstore_uri = upload_vectorstore_to_s3(str(persist_dir), bucket)

        if persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)
        if tmp_pdf_path.exists():
            tmp_pdf_path.unlink()

        return {
            "pdf_url": pdf_url,
            "vectorstore": vectorstore_uri,
            "ocr_pages": ocr_pages,
            "raw_text": final_text,
            "refined_text": refined_text,
        }

    except Exception as exc:
        print(exc)
        return {"error": str(exc)}
