from flask import Flask, request,jsonify,Blueprint
from ocr import extract_pdf
from helpers import  clean_and_parse_json,clean_content
from helpers import get_links

uplaodpdf  = Blueprint('uplaodpdf',__name__)
@uplaodpdf.route('/upload_pdf',methods=['POST'])
def upload_pdf():
    try:
        req = request.get_json()
        userID = req.get('userID') or "demo"
        key = req.get("key") or ""
        bucket = req.get("bucket") or ""
        print(key,bucket)

        content =  extract_pdf(bucket,key)
        content = clean_content(content)

        pdf_link = get_links(content)
        if pdf_link.get("error"):
            return jsonify({"error": pdf_link["error"]}), 500

        return jsonify({"pdflink": pdf_link}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
