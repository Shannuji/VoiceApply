# VoiceApply — KriyaDocs assessment starter

A beginner-friendly local prototype for a personalised job application assistant.

## Requirements
- Python 3.10 or newer
- Internet only if you want to call the optional Gemini API
- A Gemini API key is optional; the app still runs without one, but uses a safe fallback rather than AI generation.

## Run it
1. Install Python from https://www.python.org/downloads/ if you do not have it.
2. Extract this ZIP to a folder.
3. Open Terminal / Command Prompt in that folder.
4. Create a virtual environment:

   Windows:
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

6. Optional: set your Gemini API key.

   Windows Command Prompt (current terminal only):
   ```bat
   set GEMINI_API_KEY=YOUR_KEY_HERE
   ```

   Windows PowerShell:
   ```powershell
   $env:GEMINI_API_KEY="YOUR_KEY_HERE"
   ```

   macOS/Linux:
   ```bash
   export GEMINI_API_KEY="YOUR_KEY_HERE"
   ```

   Do not paste your API key into app.py or upload it to GitHub. Check the API provider's current terms, quotas, and data policy before sending real personal documents.

7. Start the app:
   ```bash
   streamlit run app.py
   ```
8. A browser tab should open with the local application.

## Suggested demo
1. Upload a resume and one cover letter.
2. Add at least 3 previous application questions and the exact answers you wrote.
3. Add a privacy rule such as a real employer name -> "a previous employer".
4. Enter a question that is similar to one of the stored questions.
5. Show the retrieved answer references and generated draft.
6. Edit the draft and save the reviewed version.
7. Open History to show the saved draft and reviewed answer.

## Important limitations
- This is a local single-user prototype, not production-ready.
- SQLite file `voiceapply.db` is created in the project folder.
- Scanned/image-only PDFs may not yield text; OCR is not included.
- TF-IDF is lexical search and may miss semantically similar questions.
- Style analysis is a rough heuristic, not a trained personal model.
- Privacy substitutions are exact-text replacements and can miss names/identifiers.
- Without `GEMINI_API_KEY`, the app does not produce a new LLM answer; it displays a safe reference/fallback draft.
- Before public deployment, add authentication, authorization, secure secret handling, data deletion/export, retention rules, rate limits, and stronger privacy testing.
