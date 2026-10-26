
import os
import json
import logging

import fitz
from flask import Flask, render_template, request
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

api_key = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key) if api_key else None


def extract_resume_text(file):
    filename = file.filename.lower()
    content = file.read()

    if filename.endswith(".pdf"):
        with fitz.open(stream=content, filetype="pdf") as pdf:
            text = "\n".join(page.get_text() for page in pdf)
    elif filename.endswith(".txt"):
        text = content.decode("utf-8-sig")
    else:
        raise ValueError("Please upload a PDF or TXT file.")

    if not text.strip():
        raise ValueError(
            "No readable text found. Try a text-based PDF or TXT file."
        )

    return text


def analyze_resume(text, summary_style, analysis_level):
    if not client:
        raise ValueError(
            "Gemini API key is missing. Check your .env file."
        )

    prompt = f"""
You are a careful resume analyzer.

Analyze the resume text below. Only report information supported
by the resume. Never invent missing details.

Return valid JSON with exactly these keys:
{{
  "name": null,
  "email": null,
  "skills": [],
  "education": [],
  "experience": [],
  "summary": "",
  "missing_details": []
}}

Rules:
- Use null for a name or email that is not found.
- Use empty arrays when list details are not found.
- Skills must be a list of strings.
- Education must be a list of strings.
- Experience must be a list of strings.
- Summary must be a string.
- missing_details must be a list of important details
  that are absent or unclear in the resume.
- Do not include markdown fences.
- Summary style: {summary_style}
- Analysis level: {analysis_level}

RESUME TEXT:
{text[:18000]}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    result = json.loads(response.text)

    # Ensure every expected field has a safe value.
    result.setdefault("name", None)
    result.setdefault("email", None)
    result.setdefault("skills", [])
    result.setdefault("education", [])
    result.setdefault("experience", [])
    result.setdefault("summary", "")
    result.setdefault("missing_details", [])

    return result


@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    filename = None
    extracted_text = None
    analysis = None

    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        summary_style = request.form.get("summary_style", "brief")
        analysis_level = request.form.get("analysis_level", "standard")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select your resume first."

        elif not uploaded_file.filename.lower().endswith((".pdf", ".txt")):
            error = "Only PDF and TXT files are supported."

        else:
            filename = os.path.basename(uploaded_file.filename)

            try:
                extracted_text = extract_resume_text(uploaded_file)

                analysis = analyze_resume(
                    extracted_text,
                    summary_style,
                    analysis_level
                )

            except ValueError as exc:
                error = str(exc)
                extracted_text = None

            except Exception:
                app.logger.exception("Resume analysis failed")
                error = (
                    "Analysis failed. Check your internet connection, "
                    "Gemini API key, API quota, and terminal error."
                )
                analysis = None

    return render_template(
        "index.html",
        error=error,
        filename=filename,
        extracted_text=extracted_text,
        analysis=analysis
    )


@app.errorhandler(413)
def file_too_large(error):
    return (
        "File is too large. Please choose a file under 5 MB.",
        413
    )


if __name__ == "__main__":
    app.run(debug=True)
