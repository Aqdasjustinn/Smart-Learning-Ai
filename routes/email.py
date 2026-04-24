from flask import request, jsonify, Blueprint
from openrouter_client import prompt_completion

email = Blueprint('email',__name__)


def get_email_body(data):
    selected_email = data.get("SelectedEmail") or {}
    if isinstance(selected_email, dict):
        body = selected_email.get("body") or {}
        if isinstance(body, dict):
            return body.get("content", "")
    return str(selected_email)

summary_prompt = """
Summarize the email in simple clear bullet points. Do not use any bold, italic, or special formatting. 
Do not add explanations. Keep it short and readable.

Format:
- Main purpose of the email
- Important details
- Any action required (or write: No action required)

Email Content:
{email_body}
"""

draft_prompt = """
Write a clean and professional reply email based on the message below.
No bold, no italics, no markdown, no emojis.
Keep sentences clear, polite, and concise.
If the email requests something, acknowledge it and respond appropriately.
If the request is unclear, politely ask for clarification.

Format:
Subject: (suggest a clear subject based on context after that next line) 
Message:
(Write the reply email in plain text paragraphs start from the next line)

Email to Respond To:
{email_body}

Now write the reply:
"""

events_prompt = """
Extract any mentioned dates, times, meetings, deadlines, or events from the email. 
Output only plain text. No bold, no markdown. No extra commentary.

Format:
Event/Meeting Name (if clear, otherwise leave blank)
Date:
Time:
Location:
Action Required:
----

If there are no events or dates mentioned, output:
No events or dates found.

Email Content:
{email_body}
"""


@email.route('/summarize_email', methods=['POST'])
def summarize_email():
    try:
        data = request.get_json()
        email_content = get_email_body(data)
        prompt = summary_prompt.format(email_body=email_content)
        summary = prompt_completion(prompt, temperature=0.3)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@email.route('/draftmail', methods=['POST'])
def draft_mail():
    try:
        data = request.get_json()
        email_body = get_email_body(data)

        draftmail = prompt_completion(draft_prompt.format(email_body=email_body), temperature=0.5)

        return jsonify({"draftmail": draftmail}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@email.route('/getevents', methods=['POST'])
def extract_events():
    try:
        data = request.get_json()
        email_content = get_email_body(data)
        prompt = events_prompt.format(email_body=email_content)
        events = prompt_completion(prompt, temperature=0.2)
        return jsonify({"events": events}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
