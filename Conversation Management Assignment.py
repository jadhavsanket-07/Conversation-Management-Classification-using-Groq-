# Imports

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import re
import sqlite3

# API Key

GROQ_API_KEY = "gsk_JSv4EVniixnlgxAVETzxWGdyb3FYiJ9NmeofeuOQYfLBi1St5"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

print("API Key Is Set. Using Model:", GROQ_MODEL)

# Chat helper


def groq_chat(messages: List[Dict[str, str]],
              functions: Optional[List[Dict[str, Any]]] = None,
              function_call: Optional[Dict[str, str]] = None,
              max_tokens: int = 512,
              temperature: float = 0.0,
              model: Optional[str] = None
              ) -> Dict[str, Any]:
    model = model or GROQ_MODEL
    url = f"{GROQ_API_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if functions is not None:
        payload["functions"] = functions
    if function_call is not None:
        payload["function_call"] = function_call

    resp = requests.post(url, headers=HEADERS, data=json.dumps(payload))
    if resp.status_code == 401:
        raise RuntimeError("401 Unauthorized: Check your GROQ_API_KEY")
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} error: {resp.text[:500]}")
    return resp.json()

# Conversation


class ConversationManager:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.run_counter = 0

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "ts": datetime.now(timezone.utc).isoformat()
        })
        self.run_counter += 1

    def get_messages(self):
        return [{"role": h["role"], "content": h["content"]} for h in self.history]

    def truncate_by_turns(self, n_turns: int):
        return self.get_messages()[-n_turns:] if n_turns > 0 else []

    def truncate_by_chars(self, max_chars: int):
        kept, total = [], 0
        for h in reversed(self.history):
            l = len(h["content"])
            if total + l > max_chars and kept:
                break
            kept.insert(0, {"role": h["role"], "content": h["content"]})
            total += l
        return kept

    def truncate_by_words(self, max_words: int):
        kept, total = [], 0
        for h in reversed(self.history):
            l = len(h["content"].split())
            if total + l > max_words and kept:
                break
            kept.insert(0, {"role": h["role"], "content": h["content"]})
            total += l
        return kept

    def summarize_history(self, summarization_instructions="Provide a concise summary + 3 bullets + 1 line summary.",
                          replace=True, max_tokens=400):
        if not self.history:
            return {"summary": "", "raw_resp": None}
        combined = "\n\n".join(
            f"{h['role'].upper()}: {h['content']}" for h in self.history)
        messages = [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": f"Summarize this conversation. {summarization_instructions}\n\n{combined}"}
        ]
        resp = groq_chat(messages, max_tokens=max_tokens)
        out_text = resp.get("choices", [{}])[0].get(
            "message", {}).get("content", "")
        if replace:
            self.history = [{"role": "assistant", "content": out_text,
                             "ts": datetime.now(timezone.utc).isoformat()}]
        return {"summary": out_text, "raw_resp": resp}

    def periodic_summarize_check(self, k: int, summarization_instructions="Concise summary.", replace=True):
        if k > 0 and self.run_counter % k == 0:
            return self.summarize_history(summarization_instructions, replace=replace)
        return None


# JSON Schema & Validation
EXTRACTION_SCHEMA = {
    "name": {"type": "string"},
    "email": {"type": "string"},
    "phone": {"type": "string"},
    "location": {"type": "string"},
    "age": {"type": "string"}
}

extract_function = {
    "name": "extract_user_info",
    "description": "Extract name, email, phone, location, and age.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "location": {"type": "string"},
            "age": {"type": "string"}
        }
    }
}


def parse_function_call_response(resp_json):
    msg = resp_json.get("choices", [{}])[0].get("message", {})
    fc = msg.get("function_call")
    if not fc:
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            fc = tool_calls[0].get("function", {})
    if fc:
        try:
            return json.loads(fc.get("arguments", "{}")), resp_json
        except:
            return {}, resp_json
    try:
        return json.loads(msg.get("content", "{}")), resp_json
    except:
        return {}, resp_json


_email_re = re.compile(r"[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+")
_phone_re = re.compile(r"[\d\+\-\s\(\)]{7,20}")


def validate_extraction(parsed):
    errors, norm = [], {}
    for k, spec in EXTRACTION_SCHEMA.items():
        val = parsed.get(k)
        if k == "age":
            try:
                norm[k] = int(val) if val is not None else None
            except:
                errors.append("age invalid int")
                norm[k] = None
        else:
            norm[k] = str(val) if val else None
            if k == "email" and norm[k] and not _email_re.search(norm[k]):
                errors.append("bad email")
            if k == "phone" and norm[k] and not _phone_re.search(norm[k]):
                errors.append("bad phone")
    return (len(errors) == 0), errors, norm

# Extraction


def extract_with_retry(chat_text: str):
    messages = [
        {"role": "system", "content": "Extract JSON info."},
        {"role": "user", "content": chat_text}
    ]
    resp = groq_chat(messages, functions=[extract_function],
                     function_call={"name": "extract_user_info"}, max_tokens=200)
    parsed, _ = parse_function_call_response(resp)
    valid, errs, norm = validate_extraction(parsed)

    if "age invalid int" in errs:
        retry_messages = messages + \
            [{"role": "system", "content": "Ensure age is returned as an integer only."}]
        resp = groq_chat(retry_messages, functions=[extract_function],
                         function_call={"name": "extract_user_info"}, max_tokens=200)
        parsed, _ = parse_function_call_response(resp)
        valid, errs, norm = validate_extraction(parsed)
    return norm, errs

# SQLite Save


def init_db():
    conn = sqlite3.connect("conversations.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT, phone TEXT, location TEXT, age INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(record: Dict[str, Any]):
    conn = sqlite3.connect("conversations.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name,email,phone,location,age) VALUES (?,?,?,?,?)",
                (record.get("name"), record.get("email"), record.get("phone"),
                 record.get("location"), record.get("age")))
    conn.commit()
    conn.close()


# Demo Run
if __name__ == "__main__":
    print("\n Task 1: Conversation Management & Summarization")
    cm = ConversationManager()

    demo_msgs = [
        ("user", "Hi, can you help me with my booking?"),
        ("assistant", "Sure, please provide your booking ID."),
        ("user", "The ID is 10257."),
        ("assistant", "Thanks. I see your booking for tomorrow at 11 AM."),
        ("user", "Can I reschedule it to next Monday?"),
    ]

    for role, content in demo_msgs:
        cm.add_message(role, content)
        summary = cm.periodic_summarize_check(
            k=3, summarization_instructions="Summarize briefly.", replace=False)
        if summary:
            print("\n[Summary After 3rd Message]:")
            print(summary["summary"])

    # Show truncations
    print("\nLast 2 turns:")
    print(cm.truncate_by_turns(2))
    print("\nTruncate by 50 chars:")
    print(cm.truncate_by_chars(50))
    print("\nTruncate by 15 words:")
    print(cm.truncate_by_words(15))

    # Final summarization
    print("\nFinal Summary of Conversation:")
    print(cm.summarize_history()["summary"])

    print("\n Task 2: JSON Extraction & Classification")
    init_db()

    sample_chats = [
        "Hello, my name is Saket Jadhav. My email is saketjadhav25@gmail.com, phone +91 7020328045. I'm 25 from Pune.",
        "Hey — I'm Rohit. rohit_patel123@gmail.com. I moved to Bangalore. Call me at 9876543210.",
        "This is Priya. Age: 31. Email: priya@example.co.in. Location: Chennai."
    ]

    for i, chat in enumerate(sample_chats, 1):
        print(f"\n=== Chat {i} ===")
        norm, errs = extract_with_retry(chat)
        print("Extracted:", norm)
        print("Errors:", errs)
        save_to_db(norm)

    print("\n Data Saved Into Conversations.db (Table: Users)")

