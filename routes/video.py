import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import whisper
import yt_dlp
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from helpers import get_links
from redis_cache import CACHE_TTL_SECONDS, get_json, set_json
from whisper.audio import SAMPLE_RATE

whisper_bp = Blueprint("whisper", __name__)

AUDIO_DIR = Path("audio")
UPLOAD_DIR = Path("uploads")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
DEVICE = "cpu"


def resolve_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_path)
        return ffmpeg_path
    except Exception:
        return None


FFMPEG_PATH = resolve_ffmpeg()
model = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
results = {}


def require_ffmpeg():
    if FFMPEG_PATH or shutil.which("ffmpeg"):
        return

    raise RuntimeError(
        "ffmpeg is required for local transcription. Install ffmpeg or add imageio-ffmpeg to the environment."
    )


def set_job_error(job_id, message):
    payload = {"error": message}
    results[job_id] = payload
    set_json(f"video_job:{job_id}", payload, CACHE_TTL_SECONDS)


def normalize_notes_payload(payload):
    if isinstance(payload, tuple):
        payload = payload[0]

    if not isinstance(payload, dict):
        raise RuntimeError("Notes generation returned an unexpected response format.")

    pdf_url = payload.get("pdf_url")
    vectorstore = payload.get("vectorstore")
    if not pdf_url or not vectorstore:
        raise RuntimeError("Notes generation did not return both pdf_url and vectorstore.")

    return {"pdf_url": pdf_url, "vectorstore": vectorstore}


def finalize_transcript(text, job_id):
    notes_payload = normalize_notes_payload(get_links(text))
    results[job_id] = notes_payload
    set_json(f"video_job:{job_id}", notes_payload, CACHE_TTL_SECONDS)
    return notes_payload


def load_audio_with_ffmpeg(media_path):
    require_ffmpeg()

    command = [
        FFMPEG_PATH,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(media_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]

    try:
        completed = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.decode("utf-8", errors="ignore").strip() or str(exc)
        raise RuntimeError(f"ffmpeg audio conversion failed: {error_message}") from exc

    audio = np.frombuffer(completed.stdout, np.int16).flatten().astype(np.float32) / 32768.0
    return audio


def transcribe_media_file(media_path):
    audio = load_audio_with_ffmpeg(media_path)
    if audio.size == 0:
        raise RuntimeError("No audio could be extracted from the media file.")

    result = model.transcribe(audio, fp16=False)
    return result["text"]


def process_uploaded_video(file_path, job_id):
    try:
        transcript = transcribe_media_file(file_path)
        finalize_transcript(transcript, job_id)
    except Exception as exc:
        set_job_error(job_id, f"Video processing failed: {exc}")
    finally:
        if file_path.exists():
            file_path.unlink()


def build_yt_dlp_options():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "ffmpeg_location": FFMPEG_PATH,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    node_path = shutil.which("node")
    if node_path:
        options["js_runtimes"] = {"node": {"path": node_path}}

    return options


def process_youtube_video(url, job_id):
    audio_path = None

    try:
        require_ffmpeg()

        with yt_dlp.YoutubeDL(build_yt_dlp_options()) as ydl:
            info = ydl.extract_info(url, download=True)
            prepared_path = Path(ydl.prepare_filename(info))

        if prepared_path.exists():
            audio_path = prepared_path
        else:
            video_id = info.get("id", "")
            matches = sorted(AUDIO_DIR.glob(f"{video_id}.*"))
            if not matches:
                raise FileNotFoundError("Downloaded YouTube audio file was not found.")
            audio_path = matches[0]

        transcript = transcribe_media_file(audio_path)
        finalize_transcript(transcript, job_id)
    except Exception as exc:
        set_job_error(job_id, f"YouTube processing failed: {exc}")
    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink()


@whisper_bp.route("/upload_video", methods=["POST"])
def upload_video():
    try:
        video_file = request.files.get("video")
        if not video_file:
            return jsonify({"error": "No file uploaded"}), 400

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        original_name = secure_filename(video_file.filename or "video")
        suffix = Path(original_name).suffix or ".mp4"
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
        video_file.save(file_path)

        job_id = str(int(time.time() * 1000))
        threading.Thread(target=process_uploaded_video, args=(file_path, job_id), daemon=True).start()
        return jsonify({"job_id": job_id, "device": DEVICE}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@whisper_bp.route("/upload_yt", methods=["POST"])
def upload_yt():
    try:
        req = request.get_json(force=True)
        url = (req.get("video") or "").strip()
        if not url:
            return jsonify({"error": "Missing YouTube URL"}), 400

        job_id = str(int(time.time() * 1000))
        threading.Thread(target=process_youtube_video, args=(url, job_id), daemon=True).start()
        return jsonify({"job_id": job_id, "device": DEVICE}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@whisper_bp.route("/get_result/<job_id>", methods=["GET"])
def get_result(job_id):
    if job_id in results:
        return jsonify(results[job_id]), 200

    cached_result = get_json(f"video_job:{job_id}")
    if cached_result is not None:
        results[job_id] = cached_result
        return jsonify(cached_result), 200

    return jsonify({"status": "processing"}), 202
