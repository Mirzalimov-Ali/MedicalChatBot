"""Foydalanuvchilar, API kalitlar va sarf (usage) — frontier API'lar (OpenAI, Anthropic) kabi. SQLite, faqat stdlib.

Oqim:
  1. Foydalanuvchi UI'da email + parol bilan ro'yxatdan o'tadi -> sessiya tokeni (sess_...) oladi.
  2. "API kalitlar" menyusida kalit yaratadi: sk-th-... faqat BIR MARTA ko'rsatiladi, bazada esa uning sha256 xeshi
     va qisqa ko'rinishi (sk-th-abc…wxyz) turadi. Baza o'g'irlansa ham kalitlarning o'zi chiqib ketmaydi.
  3. Dasturlar /v1/chat/completions ga "Authorization: Bearer sk-th-..." bilan so'rov yuboradi; har so'rov uchun
     token sarfi yoziladi, minutiga so'rovlar soni cheklanadi (RATE_LIMIT), kalitni istalgan payt bekor qilish mumkin.

Parollar scrypt bilan xeshlanadi (sekin xesh: parolni taxmin qilish qimmat). Kalitlar uzun tasodifiy satr, ularga
tez sha256 yetarli. Muhit o'zgaruvchilari: DB_PATH (baza fayli), RATE_LIMIT (so'rov/minut), SIGNUP_CODE (taklif kodi:
berilsa, faqat shu kodni bilganlar ro'yxatdan o'ta oladi — havola ochiq internetda bo'lgani uchun).
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent / "medicalai.db"))
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "20"))   # bitta kalit uchun so'rov/minut
SIGNUP_CODE = os.environ.get("SIGNUP_CODE", "")
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "500"))   # bitta foydalanuvchi uchun API so'rov/kun (admin o'zgartira oladi)
# Admin panel: shu emaillar bilan kirganlar hamma foydalanuvchilar, sarf va bloklashni ko'radi
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}
SESSION_DAYS = 30
MAX_KEYS = 10

PRICES = {
    "sft-qlora": (0.0, 0.0),
    "smollm2-1.7b-instruct": (0.0, 0.0),
}

EXPIRY_DAYS = (7, 30, 60, 90, 365)   # kalit muddati tanlovlari; None = muddatsiz

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE IF NOT EXISTS users    (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL,
                                     created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS api_keys (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL,
                                     key_hash TEXT UNIQUE NOT NULL, preview TEXT NOT NULL, created REAL NOT NULL,
                                     last_used REAL, revoked REAL);
CREATE TABLE IF NOT EXISTS usage    (id INTEGER PRIMARY KEY, key_id INTEGER NOT NULL, ts REAL NOT NULL,
                                     model TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
                                     completion_tokens INTEGER NOT NULL);
""")
# Keyin qo'shilgan ustunlar: eski bazalar (Kaggle'dan HF orqali tiklangan) ham ishlashi uchun ALTER bilan
for table, column, kind in [("users", "blocked", "REAL"), ("users", "daily_limit", "INTEGER"), ("usage", "ms", "INTEGER"),
                             ("api_keys", "model", "TEXT"), ("api_keys", "expires", "REAL"), ("usage", "cost", "REAL")]:
    # api_keys.model: NULL = barcha modellar; api_keys.expires: NULL = muddatsiz; usage.cost: dollarda
    if column not in [r[1] for r in db.execute(f"PRAGMA table_info({table})")]:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
db.commit()
lock = threading.Lock()   # sqlite ulanishi bir nechta oqimda: yozishlar navbat bilan


def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def hash_password(password):
    salt = secrets.token_bytes(16)
    return "scrypt$" + salt.hex() + "$" + hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1).hex()


def check_password(password, stored):
    _, salt, digest = stored.split("$")
    new = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1).hex()
    return hmac.compare_digest(new, digest)


def query(sql, args=(), one=False):
    with lock:
        cur = db.execute(sql, args)
        db.commit()
        rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


# ---------- sessiyalar (UI uchun) ----------

def new_session(user_id):
    token = "sess_" + secrets.token_urlsafe(32)
    query("INSERT INTO sessions VALUES (?, ?, ?)", (sha256(token), user_id, time.time() + SESSION_DAYS * 86400))
    return token


def current_user(authorization):
    """UI so'rovlari: Authorization: Bearer sess_... -> foydalanuvchi, aks holda 401."""
    token = (authorization or "").removeprefix("Bearer ").strip()
    row = query("SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires > ?",
                (sha256(token), time.time()), one=True)
    if row is None:
        raise HTTPException(401, "Avval tizimga kiring")
    if row["blocked"]:
        raise HTTPException(403, "Hisobingiz bloklangan")
    return row


def current_admin(authorization):
    user = current_user(authorization)
    if user["email"] not in ADMIN_EMAILS:
        raise HTTPException(403, "Only for admin")
    return user


# ---------- API kalitlar (dasturlar uchun) ----------

class ApiError(Exception):
    """/v1 xatolari: app.py ularni OpenAI formatida qaytaradi ({"error": {...}})."""

    def __init__(self, status, message, code):
        self.status, self.message, self.code = status, message, code


hits = defaultdict(deque)   # key_id -> oxirgi 60 soniyadagi so'rovlar vaqti (xotirada, qayta ishga tushsa nolga)


def check_api_key(authorization, x_api_key):
    """Kalitni tekshiradi va rate limit qo'llaydi. OpenAI uslubi (Bearer) ham, Anthropic uslubi (x-api-key) ham."""
    secret = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
    if not secret:
        raise ApiError(401, "API kalit berilmagan: 'Authorization: Bearer sk-th-...' sarlavhasini qo'shing", "missing_api_key")
    row = query("""SELECT k.*, u.blocked, COALESCE(u.daily_limit, ?) AS daily_limit FROM api_keys k
                   JOIN users u ON u.id = k.user_id WHERE k.key_hash = ? AND k.revoked IS NULL""",
                (DAILY_LIMIT, sha256(secret)), one=True)
    if row is None:
        raise ApiError(401, "Noto'g'ri yoki bekor qilingan API kalit", "invalid_api_key")
    if row["expires"] and row["expires"] < time.time():
        raise ApiError(401, "API kalitning muddati tugagan: yangi kalit yarating", "api_key_expired")
    if row["blocked"]:
        raise ApiError(403, "Hisob admin tomonidan bloklangan", "account_blocked")
    now = time.time()
    today = query("""SELECT COUNT(*) FROM usage u JOIN api_keys k ON k.id = u.key_id
                     WHERE k.user_id = ? AND u.ts > ?""", (row["user_id"], now - 86400), one=True)[0]
    if today >= row["daily_limit"]:
        raise ApiError(429, f"Kunlik limit: 24 soatda {row['daily_limit']} ta so'rov", "daily_limit_exceeded")
    q = hits[row["id"]]
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise ApiError(429, f"Rate limit: minutiga {RATE_LIMIT} ta so'rov. {int(q[0] + 60 - now) + 1} s kuting",
                        "rate_limit_exceeded")
    q.append(now)
    query("UPDATE api_keys SET last_used = ? WHERE id = ?", (now, row["id"]))
    return row, RATE_LIMIT - len(q)


def check_model(key, model):
    """Kalit bitta modelga bog'langan bo'lsa, boshqa modelni chaqirish mumkin emas."""
    if key["model"] and key["model"] != model:
        raise ApiError(403, f"Bu kalit faqat '{key['model']}' modeli uchun", "model_not_allowed")


def cost_of(model, prompt_tokens, completion_tokens):
    price_in, price_out = PRICES.get(model, (0, 0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def record_usage(key_id, model, prompt_tokens, completion_tokens, ms=None):
    query("""INSERT INTO usage (key_id, ts, model, prompt_tokens, completion_tokens, ms, cost)
             VALUES (?, ?, ?, ?, ?, ?, ?)""",
          (key_id, time.time(), model, prompt_tokens, completion_tokens, ms, cost_of(model, prompt_tokens, completion_tokens)))


# ---------- UI uchun endpointlar: /auth/*, /keys, /usage ----------

router = APIRouter()


class Credentials(BaseModel):
    email: str
    password: str
    code: str = ""   # SIGNUP_CODE berilgan bo'lsa kerak


class NewKey(BaseModel):
    name: str = ""
    model: str | None = None   # None = barcha modellar
    expires_days: int | None = None   # None = muddatsiz


def public(user):
    return {"id": user["id"], "email": user["email"], "created": user["created"], "is_admin": user["email"] in ADMIN_EMAILS}


@router.get("/auth/config")
def auth_config():
    return {"signup_code_required": bool(SIGNUP_CODE), "rate_limit": RATE_LIMIT}


@router.post("/auth/signup")
def signup(c: Credentials):
    email = c.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(400, "Email noto'g'ri")
    if len(c.password) < 8:
        raise HTTPException(400, "Parol kamida 8 belgi bo'lsin")
    if SIGNUP_CODE and not hmac.compare_digest(c.code.strip(), SIGNUP_CODE):
        raise HTTPException(403, "Taklif kodi noto'g'ri")
    try:
        query("INSERT INTO users (email, pw_hash, created) VALUES (?, ?, ?)", (email, hash_password(c.password), time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Bu email bilan hisob allaqachon bor")
    user = query("SELECT * FROM users WHERE email = ?", (email,), one=True)
    return {"token": new_session(user["id"]), "user": public(user)}


@router.post("/auth/login")
def login(c: Credentials):
    user = query("SELECT * FROM users WHERE email = ?", (c.email.strip().lower(),), one=True)
    if user is None or not check_password(c.password, user["pw_hash"]):
        raise HTTPException(401, "Email yoki parol noto'g'ri")
    if user["blocked"]:
        raise HTTPException(403, "Hisobingiz bloklangan")
    return {"token": new_session(user["id"]), "user": public(user)}


@router.post("/auth/logout")
def logout(authorization: str = Header("")):
    query("DELETE FROM sessions WHERE token_hash = ?", (sha256(authorization.removeprefix("Bearer ").strip()),))
    return {"ok": True}


@router.get("/auth/me")
def me(authorization: str = Header("")):
    return public(current_user(authorization))


@router.get("/keys")
def list_keys(authorization: str = Header("")):
    user = current_user(authorization)
    rows = query("""SELECT k.id, k.name, k.preview, k.model, k.created, k.expires, k.last_used,
                           COUNT(u.id) AS requests, COALESCE(SUM(u.prompt_tokens + u.completion_tokens), 0) AS tokens,
                           COALESCE(SUM(u.cost), 0) AS cost
                    FROM api_keys k LEFT JOIN usage u ON u.key_id = k.id
                    WHERE k.user_id = ? AND k.revoked IS NULL GROUP BY k.id ORDER BY k.created DESC""", (user["id"],))
    return [dict(r) for r in rows]


@router.post("/keys")
def create_key(body: NewKey, authorization: str = Header("")):
    user = current_user(authorization)
    if len(list_keys(authorization)) >= MAX_KEYS:
        raise HTTPException(400, f"Ko'pi bilan {MAX_KEYS} ta faol kalit")
    secret = "sk-th-" + secrets.token_urlsafe(32)
    name = body.name.strip()[:60] or "Mening kalitim"
    model = (body.model or "").strip()[:60] or None   # bo'sh = barcha modellar
    if body.expires_days is not None and body.expires_days not in EXPIRY_DAYS:
        raise HTTPException(400, f"Muddat: {', '.join(map(str, EXPIRY_DAYS))} kun yoki muddatsiz")
    now = time.time()
    expires = now + body.expires_days * 86400 if body.expires_days else None
    query("INSERT INTO api_keys (user_id, name, key_hash, preview, model, created, expires) VALUES (?, ?, ?, ?, ?, ?, ?)",
          (user["id"], name, sha256(secret), secret[:10] + "…" + secret[-4:], model, now, expires))
    # Kalitning o'zi faqat shu javobda: keyin uni hech kim (server ham) ko'rsata olmaydi.
    return {"key": secret, "name": name, "model": model, "expires": expires, "preview": secret[:10] + "…" + secret[-4:]}


@router.delete("/keys/{key_id}")
def revoke_key(key_id: int, authorization: str = Header("")):
    user = current_user(authorization)
    query("UPDATE api_keys SET revoked = ? WHERE id = ? AND user_id = ? AND revoked IS NULL",
          (time.time(), key_id, user["id"]))
    hits.pop(key_id, None)
    return {"ok": True}


@router.get("/usage")
def usage(authorization: str = Header(""), days: int = 30):
    """Oxirgi `days` kundagi sarf, kunlar bo'yicha (UI'dagi "Sarf" jadvali)."""
    user = current_user(authorization)
    rows = query("""SELECT date(u.ts, 'unixepoch') AS day, COUNT(*) AS requests,
                           SUM(u.prompt_tokens) AS prompt_tokens, SUM(u.completion_tokens) AS completion_tokens,
                           COALESCE(SUM(u.cost), 0) AS cost
                    FROM usage u JOIN api_keys k ON k.id = u.key_id
                    WHERE k.user_id = ? AND u.ts > ? GROUP BY day ORDER BY day DESC""",
                 (user["id"], time.time() - days * 86400))
    return [dict(r) for r in rows]


# ---------- admin panel: /admin/* (only ADMIN_EMAILS) ----------

class UserUpdate(BaseModel):
    blocked: bool | None = None
    daily_limit: int | None = None   # 0 yoki manfiy -> standart DAILY_LIMIT


@router.get("/admin/stats")
def admin_stats(authorization: str = Header(""), days: int = 30):
    """Butun platforma: jami raqamlar, kunlik grafik, har bir foydalanuvchi qancha ishlatgani."""
    current_admin(authorization)
    since = time.time() - days * 86400
    totals = query("""SELECT (SELECT COUNT(*) FROM users) AS users,
                             (SELECT COUNT(*) FROM api_keys WHERE revoked IS NULL) AS active_keys,
                             COUNT(*) AS requests, COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                             COALESCE(SUM(completion_tokens), 0) AS completion_tokens, CAST(AVG(ms) AS INTEGER) AS avg_ms
                      FROM usage WHERE ts > ?""", (since,), one=True)
    daily = query("""SELECT date(ts, 'unixepoch') AS day, COUNT(*) AS requests,
                            SUM(prompt_tokens + completion_tokens) AS tokens, CAST(AVG(ms) AS INTEGER) AS avg_ms
                     FROM usage WHERE ts > ? GROUP BY day ORDER BY day""", (since,))
    users = query("""SELECT u.id, u.email, u.created, u.blocked, COALESCE(u.daily_limit, ?) AS daily_limit,
                            (SELECT COUNT(*) FROM api_keys k WHERE k.user_id = u.id AND k.revoked IS NULL) AS keys,
                            COUNT(g.id) AS requests, COALESCE(SUM(g.prompt_tokens + g.completion_tokens), 0) AS tokens,
                            MAX(g.ts) AS last_request
                     FROM users u LEFT JOIN api_keys k ON k.user_id = u.id
                     LEFT JOIN usage g ON g.key_id = k.id AND g.ts > ?
                     GROUP BY u.id ORDER BY tokens DESC, u.created DESC""", (DAILY_LIMIT, since))
    return {"days": days, "daily_limit": DAILY_LIMIT, "rate_limit": RATE_LIMIT, "totals": dict(totals),
            "daily": [dict(r) for r in daily], "users": [dict(r) for r in users]}


@router.get("/admin/users/{user_id}/keys")
def admin_user_keys(user_id: int, authorization: str = Header("")):
    current_admin(authorization)
    rows = query("""SELECT k.id, k.name, k.preview, k.model, k.created, k.expires, k.last_used, k.revoked,
                           COUNT(g.id) AS requests, COALESCE(SUM(g.prompt_tokens + g.completion_tokens), 0) AS tokens,
                           COALESCE(SUM(g.cost), 0) AS cost
                    FROM api_keys k LEFT JOIN usage g ON g.key_id = k.id
                    WHERE k.user_id = ? GROUP BY k.id ORDER BY k.created DESC""", (user_id,))
    return [dict(r) for r in rows]


@router.patch("/admin/users/{user_id}")
def admin_update_user(user_id: int, body: UserUpdate, authorization: str = Header("")):
    admin = current_admin(authorization)
    if body.blocked is not None:
        if user_id == admin["id"]:
            raise HTTPException(400, "O'zingizni bloklay olmaysiz")
        query("UPDATE users SET blocked = ? WHERE id = ?", (time.time() if body.blocked else None, user_id))
        if body.blocked:   # bloklangan foydalanuvchi UI'dan ham darhol chiqariladi
            query("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    if body.daily_limit is not None:
        query("UPDATE users SET daily_limit = ? WHERE id = ?", (body.daily_limit if body.daily_limit > 0 else None, user_id))
    return {"ok": True}


@router.delete("/admin/keys/{key_id}")
def admin_revoke_key(key_id: int, authorization: str = Header("")):
    current_admin(authorization)
    query("UPDATE api_keys SET revoked = ? WHERE id = ? AND revoked IS NULL", (time.time(), key_id))
    hits.pop(key_id, None)
    return {"ok": True}