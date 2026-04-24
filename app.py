from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from utils import build_qa_chain, generate_streamlit_project,  generate_question, generate_summary, extract_events, get_topics, load_vector_store, process_pdf_rag
from helpers import  clean_and_parse_json,clean_content
from ocr import extract_pdf,run_vision
from routes.video import whisper_bp

from routes.email import email
from routes.uplaod_pdf import uplaodpdf
from routes.pdf_comp import pdf_comp
from routes.finetune import finetune
from routes.vision import vision
app = Flask(__name__)
CORS(app) 

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 
app.register_blueprint(whisper_bp)
app.register_blueprint(email)
app.register_blueprint(uplaodpdf)
app.register_blueprint(pdf_comp)
app.register_blueprint(finetune)
app.register_blueprint(vision)

# genrate ppt
@app.route('/generate_questions',methods=['POST'])
def generate_ques():
    try:
        req = request.get_json()
        prompt = req.get('prompt') or ""
        key = req.get("key") or ""
        bucket = req.get("bucket") or ""
        # print(prompt)

        if(key == ""):
            question =  generate_question(prompt,"no content")
            question =  clean_and_parse_json(question)
            return jsonify({
            "status": "success",
            "questions":question,
            }), 200
        

        content =  extract_pdf(bucket,key)
        content = clean_content(content)
        question =  generate_question(prompt,content)
        question =  clean_and_parse_json(question)
        return jsonify({
            "status": "success",
            "questions":question,
        }), 200

    except Exception as e:
        print(e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    

    
if __name__ == '__main__':
    debug_enabled = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=debug_enabled,
    )
