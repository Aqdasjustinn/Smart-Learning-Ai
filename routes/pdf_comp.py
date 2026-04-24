from flask import Flask, request,jsonify,Blueprint
import os
from dotenv import load_dotenv
from openrouter_client import prompt_completion
from utils import load_vector_store
from helpers import clean_and_parse_json
import json

pdf_comp = Blueprint('pdf_comp', __name__)
load_dotenv()

def ask_openrouter(question, context):
    prompt = f"""
   Use the PDF context to answer. 
    Context:
    {context}

    Question:
    {question}

    Answer clearly and simply.
    """
    return prompt_completion(prompt, temperature=0.3)

@pdf_comp.route('/askquestion',methods=['POST'])
def generate_ques():
    try:
        req = request.get_json()
        question = req.get('question') or ""
        persist_dir = req.get("store") or ""
        # print(persist_dir)

        if(persist_dir == ""):
            return jsonify({"message":"persist_dir is required"}), 400
        
        retriever = load_vector_store(persist_dir)
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        answer = ask_openrouter(question,context)
        answer =  clean_and_parse_json(answer)
        return jsonify({"answer":answer}),200
    except Exception as e:
        print(e)
        return jsonify({"message": str(e)}), 500
    
import json

def cluster_to_concepts(clustered_chunks):
    final_nodes = []
    final_edges = []

    for group_id, texts in clustered_chunks.items():
        combined_text = "\n".join(texts[:5])

        prompt = f"""
Extract key concepts and relationships from the text.

Return JSON ONLY in this exact structure:
{{
  "nodes": ["Concept1", "Concept2"],
  "edges": [
     ["Concept1", "relation", "Concept2"]
  ]
}}
TEXT:
{combined_text}
"""

        raw = prompt_completion(prompt, temperature=0.1).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(raw)
        except:
            continue

        for node in data.get("nodes", []):
            if node not in final_nodes:
                final_nodes.append(node)

        for edge in data.get("edges", []):
            if len(edge) == 3:
                final_edges.append({
                    "source": edge[0],
                    "label": edge[1],
                    "target": edge[2]
                })

    return {
        "nodes": [{"id": n, "label": n} for n in final_nodes],
        "edges": final_edges
    }


@pdf_comp.route("/conceptmap", methods=["POST"])
def concept_map_route():
    req = request.get_json()
    persist_dir = req.get("store")

    if not persist_dir:
        return jsonify({"error": "store is required"}), 400

    retriever = load_vector_store(persist_dir)
    docs = retriever.invoke("give main ideas and explanation")
    chunks = [d.page_content for d in docs]

    clusters = {0: chunks}
    result = cluster_to_concepts(clusters)

    return jsonify(result), 200
