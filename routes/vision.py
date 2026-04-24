from flask import Blueprint, request, jsonify, Response 
from ocr import run_vision
vision = Blueprint("vision",__name__)

@vision.route('/vision_ocr',methods=['POST'])
def vision_ocr():
    try:
        req =  request.get_json()
        key =  req.get('key') or " "
        bucket =  req.get('bucket') or ""
        userID = req.get("userID") or ""

        if key == " " or bucket == " ":
            return jsonify({"status":"error"}),400
        
        url = run_vision(bucket,key)
        if url.get("error"):
            return jsonify({"message": url["error"]}), 500

        return jsonify({"pdflink": url}), 200
        
    except Exception as e:
        print(e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
