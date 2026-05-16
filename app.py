from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import psycopg2
import time
import json
from datetime import datetime

app = Flask(__name__)

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
DB_CONFIG = {
    "host": "postgres",
    "database": "aibot",
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASS"),
    "port": 5432
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# --------------------------------------------------
# CORS
# --------------------------------------------------
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)
# --------------------------------------------------
# INIT DB TABLE
# --------------------------------------------------
def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id SERIAL PRIMARY KEY,
            user_message TEXT,
            bot_reply TEXT,
            response_time_seconds FLOAT,
            client_ip TEXT,
            user_agent TEXT,
            origin TEXT,
            use_kb BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        cur.close()
        conn.close()

        print("DB initialized: chat_logs table ready")

    except Exception as e:
        print("DB INIT ERROR:", e)
# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "qwen2.5:3b-instruct"

LOG_FILE = "/app/logs/chat_logs.txt"

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_client_ip():
    return (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.remote_addr
    )

def trim_messages(messages, max_chars=12000):
    """
    Keep latest messages within character limit.
    """

    trimmed = []
    total = 0

    for msg in reversed(messages):
        text = f"{msg.get('role')}: {msg.get('content')}\n"

        if total + len(text) > max_chars:
            break

        trimmed.insert(0, msg)
        total += len(text)

    return trimmed

def build_conversation_text(messages):
    output = []

    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")

        output.append(f"{role}: {content}")

    return "\n\n".join(output)

# --------------------------------------------------
# DATABASE LOGGING
# --------------------------------------------------
def log_chat_to_db(
    user_message,
    bot_reply,
    duration,
    metadata=None
):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO chat_logs (
                user_message,
                bot_reply,
                response_time_seconds,
                client_ip,
                user_agent,
                origin,
                use_kb
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            user_message,
            bot_reply,
            duration,
            get_client_ip(),
            request.headers.get("User-Agent"),
            request.headers.get("Origin"),
            False
        ))

        conn.commit()

        cur.close()
        conn.close()

    except Exception as e:
        print("DB LOG ERROR:", e)

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL
    })

# --------------------------------------------------
# CHAT
# --------------------------------------------------
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():

    start_time = time.time()

    user_message = ""

    try:
        data = request.get_json(force=True)

        # --------------------------------------------------
        # INPUTS
        # --------------------------------------------------
        user_message = data.get("message", "")

        messages = data.get("messages", [])

        system_prompt = data.get(
            "system_prompt",
            "You are a helpful AI assistant."
        )

        max_context_chars = int(
            data.get("max_context_chars", 12000)
        )

        temperature = float(
            data.get("temperature", 0.7)
        )

        # --------------------------------------------------
        # ADD CURRENT MESSAGE
        # --------------------------------------------------
        if user_message:
            messages.append({
                "role": "user",
                "content": user_message
            })

        # --------------------------------------------------
        # TRIM CONTEXT
        # --------------------------------------------------
        trimmed_messages = trim_messages(
            messages,
            max_context_chars
        )

        conversation_context = build_conversation_text(
            trimmed_messages
        )

        # --------------------------------------------------
        # FINAL PROMPT
        # --------------------------------------------------
        prompt = f"""
{system_prompt}

IMPORTANT RULES:
- Format as plain text only.
- No markdown.
- No HTML.
- Continue the conversation naturally.

CONVERSATION:
{conversation_context}

ASSISTANT:
"""

        # --------------------------------------------------
        # OLLAMA REQUEST
        # --------------------------------------------------
        ollama_response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": temperature
                }
            },
            timeout=180
        )

        ollama_data = ollama_response.json()

        bot_reply = ollama_data.get(
            "response",
            ""
        ).strip()

        # --------------------------------------------------
        # DURATION
        # --------------------------------------------------
        end_time = time.time()

        duration = round(
            end_time - start_time,
            3
        )

        # --------------------------------------------------
        # LOG
        # --------------------------------------------------
        log_chat_to_db(
            user_message=user_message,
            bot_reply=bot_reply,
            duration=duration,
            metadata={
                "messages_count": len(messages)
            }
        )

        # --------------------------------------------------
        # RESPONSE
        # --------------------------------------------------
        return jsonify({

            # Main reply
            "reply": bot_reply,

            # Context info
            "context_chars": len(conversation_context),
            "messages_used": len(trimmed_messages),
            "messages_total": len(messages),

            # Raw context
            "context": conversation_context,

            # Timing
            "duration": duration,

            # Model info
            "model": MODEL,

            # Debug
            "prompt": prompt,

            # Request info
            "temperature": temperature,
            "max_context_chars": max_context_chars,

            # Metadata
            "timestamp": datetime.utcnow().isoformat(),

            # Ollama raw response
            "ollama": ollama_data
        })

    except Exception as e:

        error_msg = str(e)

        end_time = time.time()

        duration = round(
            end_time - start_time,
            3
        )

        bot_reply = f"ERROR: {error_msg}"

        log_chat_to_db(
            user_message=user_message,
            bot_reply=bot_reply,
            duration=duration
        )

        return jsonify({
            "reply": f"Error: {error_msg}",
            "duration": duration
        }), 500

# --------------------------------------------------
# GLOBAL ERROR HANDLER
# --------------------------------------------------
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({
        "reply": "Unexpected server error.",
        "error": str(e)
    }), 500


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    init_db()

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )