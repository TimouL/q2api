import os
import json
import traceback
import uuid
import time
import asyncio
import importlib.util
import random
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any, AsyncGenerator, Tuple

from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import tiktoken

from gitstore_sync import build_gitstore_sync
from log_capture import enable_capture, disable_capture, capture_status, get_logs, add_structured_log

from db import init_db, close_db, row_to_dict, SQLiteBackend, PostgresBackend, MySQLBackend

# Request logging switch (in-memory, resets on app restart)
REQUEST_LOGGING_ENABLED = False

# ------------------------------------------------------------------------------
# In-memory request statistics (not persisted)
# ------------------------------------------------------------------------------
class RequestStats:
    def __init__(self):
        self.total = 0
        self.success = 0
        self.error = 0
        self.by_model: Dict[str, Dict[str, int]] = {}
        self.by_account: Dict[str, Dict[str, int]] = {}  # account_id -> {"success": x, "error": y}
    
    def record_success(self, model: str = "unknown", account_id: str = ""):
        self.total += 1
        self.success += 1
        if model not in self.by_model:
            self.by_model[model] = {"total": 0, "success": 0, "error": 0}
        self.by_model[model]["total"] += 1
        self.by_model[model]["success"] += 1
        if account_id:
            if account_id not in self.by_account:
                self.by_account[account_id] = {"success": 0, "error": 0}
            self.by_account[account_id]["success"] += 1
    
    def record_error(self, model: str = "unknown", account_id: str = ""):
        self.total += 1
        self.error += 1
        if model not in self.by_model:
            self.by_model[model] = {"total": 0, "success": 0, "error": 0}
        self.by_model[model]["total"] += 1
        self.by_model[model]["error"] += 1
        if account_id:
            if account_id not in self.by_account:
                self.by_account[account_id] = {"success": 0, "error": 0}
            self.by_account[account_id]["error"] += 1
    
    def get_account_stats(self, account_id: str) -> Dict[str, int]:
        return self.by_account.get(account_id, {"success": 0, "error": 0})
    
    def get(self) -> Dict[str, Any]:
        return {"total": self.total, "success": self.success, "error": self.error, "by_model": self.by_model}

REQUEST_STATS = RequestStats()

# ------------------------------------------------------------------------------
# In-memory account throttle tracker (not persisted, resets on app restart)
# ------------------------------------------------------------------------------
class AccountThrottleTracker:
    def __init__(self):
        self.throttled_until: Dict[str, float] = {}  # account_id -> timestamp
        self.error_counts: Dict[str, int] = {}  # account_id -> count
    
    def throttle(self, account_id: str, base_backoff: float = 30.0, max_backoff: float = 300.0) -> float:
        """Mark account as throttled with exponential backoff. Returns backoff duration."""
        error_count = self.error_counts.get(account_id, 0) + 1
        self.error_counts[account_id] = error_count
        backoff = min(base_backoff * (2 ** (error_count - 1)), max_backoff)
        self.throttled_until[account_id] = time.time() + backoff
        return backoff
    
    def is_throttled(self, account_id: str) -> bool:
        """Check if account is currently throttled."""
        until = self.throttled_until.get(account_id)
        if until is None:
            return False
        return time.time() < until
    
    def clear_throttle(self, account_id: str) -> None:
        """Clear throttle state for an account (e.g., on successful request)."""
        self.throttled_until.pop(account_id, None)
        self.error_counts.pop(account_id, None)

THROTTLE_TRACKER = AccountThrottleTracker()

# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------

try:
    # cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None

def count_tokens(text: str, apply_multiplier: bool = False) -> int:
    """Counts tokens with tiktoken."""
    if not text or not ENCODING:
        return 0
    token_count = len(ENCODING.encode(text))
    if apply_multiplier:
        token_count = int(token_count * TOKEN_COUNT_MULTIPLIER)
    return token_count

# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GITSTORE_SYNC = build_gitstore_sync(BASE_DIR)

SUPPORTED_MODELS = ["claude-sonnet-4", "claude-sonnet-4.5", "claude-haiku-4.5", "claude-opus-4.5"]

app = FastAPI(title="v2 OpenAI-compatible Server (Amazon Q Backend)")

# CORS for simple testing in browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Dynamic import of replicate.py to avoid package __init__ needs
# ------------------------------------------------------------------------------

def _load_replicate_module():
    mod_path = BASE_DIR / "replicate.py"
    spec = importlib.util.spec_from_file_location("v2_replicate", str(mod_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_replicate = _load_replicate_module()
send_chat_request = _replicate.send_chat_request

# ------------------------------------------------------------------------------
# Dynamic import of Claude modules
# ------------------------------------------------------------------------------

def _load_claude_modules():
    # claude_types
    spec_types = importlib.util.spec_from_file_location("v2_claude_types", str(BASE_DIR / "claude_types.py"))
    mod_types = importlib.util.module_from_spec(spec_types)
    spec_types.loader.exec_module(mod_types)
    
    # claude_converter
    spec_conv = importlib.util.spec_from_file_location("v2_claude_converter", str(BASE_DIR / "claude_converter.py"))
    mod_conv = importlib.util.module_from_spec(spec_conv)
    # We need to inject claude_types into converter's namespace if it uses relative imports or expects them
    # But since we used relative import in claude_converter.py (.claude_types), we need to be careful.
    # Actually, since we are loading dynamically, relative imports might fail if not in sys.modules correctly.
    # Let's patch sys.modules temporarily or just rely on file location.
    # A simpler way for this single-file script style is to just load them.
    # However, claude_converter does `from .claude_types import ...`
    # To make that work, we should probably just use standard import if v2 is a package,
    # but v2 is just a folder.
    # Let's assume the user runs this with v2 in pythonpath or we just fix imports in the files.
    # But I wrote `from .claude_types` in the file.
    # Let's try to load it. If it fails, we might need to adjust.
    # Actually, for simplicity in this `app.py` dynamic loading context,
    # it is better if `claude_converter.py` used absolute import or we mock the package.
    # BUT, let's try to just load them and see.
    # To avoid relative import issues, I will inject the module into sys.modules
    import sys
    sys.modules["v2.claude_types"] = mod_types
    
    spec_conv.loader.exec_module(mod_conv)
    
    # claude_stream
    spec_stream = importlib.util.spec_from_file_location("v2_claude_stream", str(BASE_DIR / "claude_stream.py"))
    mod_stream = importlib.util.module_from_spec(spec_stream)
    spec_stream.loader.exec_module(mod_stream)
    
    return mod_types, mod_conv, mod_stream

try:
    _claude_types, _claude_converter, _claude_stream = _load_claude_modules()
    ClaudeRequest = _claude_types.ClaudeRequest
    convert_claude_to_amazonq_request = _claude_converter.convert_claude_to_amazonq_request
    map_model_name = _claude_converter.map_model_name
    ClaudeStreamHandler = _claude_stream.ClaudeStreamHandler
except Exception as e:
    print(f"Failed to load Claude modules: {e}")
    traceback.print_exc()
    # Define dummy classes to avoid NameError on startup if loading fails
    class ClaudeRequest(BaseModel):
        pass
    convert_claude_to_amazonq_request = None
    map_model_name = lambda x: x
    ClaudeStreamHandler = None

# ------------------------------------------------------------------------------
# Global HTTP Client
# ------------------------------------------------------------------------------

GLOBAL_CLIENT: Optional[httpx.AsyncClient] = None

def _get_proxies() -> Optional[Dict[str, str]]:
    proxy = os.getenv("HTTP_PROXY", "").strip()
    if proxy:
        return {"http": proxy, "https": proxy}
    return None

async def _init_global_client():
    global GLOBAL_CLIENT
    proxies = _get_proxies()
    mounts = None
    if proxies:
        proxy_url = proxies.get("https") or proxies.get("http")
        if proxy_url:
            mounts = {
                "https://": httpx.AsyncHTTPTransport(proxy=proxy_url),
                "http://": httpx.AsyncHTTPTransport(proxy=proxy_url),
            }
    # Increased limits for high concurrency with streaming
    # max_connections: 总连接数上限
    # max_keepalive_connections: 保持活跃的连接数
    # keepalive_expiry: 连接保持时间
    limits = httpx.Limits(
        max_keepalive_connections=60,
        max_connections=60,  # 提高到500以支持更高并发
        keepalive_expiry=30.0  # 30秒后释放空闲连接
    )
    # 为流式响应设置更长的超时
    timeout = httpx.Timeout(
        connect=1.0,  # 连接超时
        read=300.0,    # 读取超时(流式响应需要更长时间)
        write=1.0,    # 写入超时
        pool=1.0      # 从连接池获取连接的超时时间(关键!)
    )
    GLOBAL_CLIENT = httpx.AsyncClient(mounts=mounts, timeout=timeout, limits=limits)

async def _close_global_client():
    global GLOBAL_CLIENT
    if GLOBAL_CLIENT:
        await GLOBAL_CLIENT.aclose()
        GLOBAL_CLIENT = None

# ------------------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------------------

# Database backend instance (initialized on startup)
_db = None

async def _ensure_db():
    """Initialize database backend."""
    global _db
    _db = await init_db()


def _db_backend_name() -> str:
    if isinstance(_db, SQLiteBackend):
        return "sqlite"
    if isinstance(_db, PostgresBackend):
        return "postgres"
    if isinstance(_db, MySQLBackend):
        return "mysql"
    return "unknown"

def _row_to_dict(r: Dict[str, Any]) -> Dict[str, Any]:
    """Convert database row to dict with JSON parsing."""
    return row_to_dict(r)

# _ensure_db() will be called in startup event

# ------------------------------------------------------------------------------
# Background token refresh thread
# ------------------------------------------------------------------------------

async def _refresh_stale_tokens():
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            if _db is None:
                print("[Error] Database not initialized, skipping token refresh cycle.")
                continue
            now = time.time()
            
            if LAZY_ACCOUNT_POOL_ENABLED:
                limit = LAZY_ACCOUNT_POOL_SIZE + LAZY_ACCOUNT_POOL_REFRESH_OFFSET
                order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
                query = f"SELECT id, last_refresh_time FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction} LIMIT {limit}"
                rows = await _db.fetchall(query)
            else:
                rows = await _db.fetchall("SELECT id, last_refresh_time FROM accounts WHERE enabled=1")

            for row in rows:
                acc_id, last_refresh = row['id'], row['last_refresh_time']
                should_refresh = False
                if not last_refresh or last_refresh == "never":
                    should_refresh = True
                else:
                    try:
                        last_time = time.mktime(time.strptime(last_refresh, "%Y-%m-%dT%H:%M:%S"))
                        if now - last_time > 1500:  # 25 minutes
                            should_refresh = True
                    except Exception:
                        # Malformed or unparsable timestamp; force refresh
                        should_refresh = True

                if should_refresh:
                    try:
                        await refresh_access_token_in_db(acc_id)
                    except Exception:
                        traceback.print_exc()
                        # Ignore per-account refresh failure; timestamp/status are recorded inside
                        pass
        except Exception:
            traceback.print_exc()
            pass

# ------------------------------------------------------------------------------
# Env and API Key authorization (keys are independent of AWS accounts)
# ------------------------------------------------------------------------------
def _parse_allowed_keys_env() -> List[str]:
    """
    OPENAI_KEYS is a comma-separated whitelist of API keys for authorization only.
    Example: OPENAI_KEYS="key1,key2,key3"
    - When the list is non-empty, incoming Authorization: Bearer {key} must be one of them.
    - When empty or unset, authorization is effectively disabled (dev mode).
    """
    s = os.getenv("OPENAI_KEYS", "") or ""
    keys: List[str] = []
    for k in [x.strip() for x in s.split(",") if x.strip()]:
        keys.append(k)
    return keys

ALLOWED_API_KEYS: List[str] = _parse_allowed_keys_env()
TOKEN_COUNT_MULTIPLIER: float = float(os.getenv("TOKEN_COUNT_MULTIPLIER", "1.0"))

# Lazy Account Pool settings
LAZY_ACCOUNT_POOL_ENABLED: bool = os.getenv("LAZY_ACCOUNT_POOL_ENABLED", "false").lower() in ("true", "1", "yes")
LAZY_ACCOUNT_POOL_SIZE: int = int(os.getenv("LAZY_ACCOUNT_POOL_SIZE", "20"))
LAZY_ACCOUNT_POOL_REFRESH_OFFSET: int = int(os.getenv("LAZY_ACCOUNT_POOL_REFRESH_OFFSET", "10"))
LAZY_ACCOUNT_POOL_ORDER_BY: str = os.getenv("LAZY_ACCOUNT_POOL_ORDER_BY", "created_at")
LAZY_ACCOUNT_POOL_ORDER_DESC: bool = os.getenv("LAZY_ACCOUNT_POOL_ORDER_DESC", "false").lower() in ("true", "1", "yes")

# Validate LAZY_ACCOUNT_POOL_ORDER_BY to prevent SQL injection
if LAZY_ACCOUNT_POOL_ORDER_BY not in ["created_at", "id", "success_count"]:
    LAZY_ACCOUNT_POOL_ORDER_BY = "created_at"

def _is_console_enabled() -> bool:
    """检查是否启用管理控制台"""
    console_env = os.getenv("ENABLE_CONSOLE", "true").strip().lower()
    return console_env not in ("false", "0", "no", "disabled")

CONSOLE_ENABLED: bool = _is_console_enabled()

# Admin authentication configuration
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")

def _extract_bearer(token_header: Optional[str]) -> Optional[str]:
    if not token_header:
        return None
    if token_header.startswith("Bearer "):
        return token_header.split(" ", 1)[1].strip()
    return token_header.strip()

async def _list_enabled_accounts(limit: Optional[int] = None, exclude_throttled: bool = True) -> List[Dict[str, Any]]:
    if LAZY_ACCOUNT_POOL_ENABLED:
        order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
        query = f"SELECT * FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction}"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    else:
        query = "SELECT * FROM accounts WHERE enabled=1 ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    
    accounts = [_row_to_dict(r) for r in rows]
    
    # Filter out throttled accounts in memory
    if exclude_throttled:
        accounts = [a for a in accounts if not THROTTLE_TRACKER.is_throttled(a["id"])]
    
    return accounts

async def _list_enabled_accounts_excluding(exclude_ids: List[str], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """List enabled accounts excluding specified IDs and throttled accounts (in-memory)."""
    if exclude_ids:
        placeholders = ",".join(["?" for _ in exclude_ids])
        exclude_condition = f"id NOT IN ({placeholders})"
        where_clause = f"enabled=1 AND {exclude_condition}"
    else:
        where_clause = "enabled=1"
    
    if LAZY_ACCOUNT_POOL_ENABLED:
        order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
        query = f"SELECT * FROM accounts WHERE {where_clause} ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction}"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query, tuple(exclude_ids) if exclude_ids else ())
    else:
        query = f"SELECT * FROM accounts WHERE {where_clause} ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query, tuple(exclude_ids) if exclude_ids else ())
    
    accounts = [_row_to_dict(r) for r in rows]
    
    # Filter out throttled accounts in memory
    accounts = [a for a in accounts if not THROTTLE_TRACKER.is_throttled(a["id"])]
    
    return accounts

async def _throttle_account(account_id: str, base_backoff: float = 30.0, max_backoff: float = 300.0) -> None:
    """Mark account as throttled with exponential backoff (in-memory only)."""
    backoff = THROTTLE_TRACKER.throttle(account_id, base_backoff, max_backoff)
    error_count = THROTTLE_TRACKER.error_counts.get(account_id, 1)
    print(f"[Throttle] Account {account_id[:8]}... throttled for {backoff:.0f}s (error_count={error_count})")

async def _list_disabled_accounts() -> List[Dict[str, Any]]:
    rows = await _db.fetchall("SELECT * FROM accounts WHERE enabled=0 ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]

async def verify_account(account: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """验证账号可用性"""
    try:
        account = await refresh_access_token_in_db(account['id'])
        test_request = {
            "conversationState": {
                "currentMessage": {"userInputMessage": {"content": "hello"}},
                "chatTriggerType": "MANUAL"
            }
        }
        _, _, tracker, event_gen = await send_chat_request(
            access_token=account['accessToken'],
            messages=[],
            stream=True,
            raw_payload=test_request
        )
        if event_gen:
            async for _ in event_gen:
                break
        return True, None
    except Exception as e:
        if "AccessDenied" in str(e) or "403" in str(e):
            return False, "AccessDenied"
        return False, None

async def resolve_account_for_key(bearer_key: Optional[str]) -> Dict[str, Any]:
    """
    Authorize request by OPENAI_KEYS (if configured), then select an AWS account.
    Selection strategy: random among all enabled accounts. Authorization key does NOT map to any account.
    """
    # Authorization
    if ALLOWED_API_KEYS:
        if not bearer_key or bearer_key not in ALLOWED_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Selection: random among enabled accounts
    if LAZY_ACCOUNT_POOL_ENABLED:
        candidates = await _list_enabled_accounts(limit=LAZY_ACCOUNT_POOL_SIZE)
    else:
        candidates = await _list_enabled_accounts()

    if not candidates:
        raise HTTPException(status_code=401, detail="No enabled account available")
    return random.choice(candidates)

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------

class AccountCreate(BaseModel):
    label: Optional[str] = None
    clientId: str
    clientSecret: str
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = True

class BatchAccountCreate(BaseModel):
    accounts: List[AccountCreate]

class AccountUpdate(BaseModel):
    label: Optional[str] = None
    clientId: Optional[str] = None
    clientSecret: Optional[str] = None
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False

# ------------------------------------------------------------------------------
# Token refresh (OIDC)
# ------------------------------------------------------------------------------

OIDC_BASE = "https://oidc.us-east-1.amazonaws.com"
TOKEN_URL = f"{OIDC_BASE}/token"

def _oidc_headers() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "user-agent": "aws-sdk-rust/1.3.9 os/windows lang/rust/1.87.0",
        "x-amz-user-agent": "aws-sdk-rust/1.3.9 ua/2.1 api/ssooidc/1.88.0 os/windows lang/rust/1.87.0 m/E app/AmazonQ-For-CLI",
        "amz-sdk-request": "attempt=1; max=3",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
    }

async def refresh_access_token_in_db(account_id: str) -> Dict[str, Any]:
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    acc = _row_to_dict(row)

    if not acc.get("clientId") or not acc.get("clientSecret") or not acc.get("refreshToken"):
        raise HTTPException(status_code=400, detail="Account missing clientId/clientSecret/refreshToken for refresh")

    payload = {
        "grantType": "refresh_token",
        "clientId": acc["clientId"],
        "clientSecret": acc["clientSecret"],
        "refreshToken": acc["refreshToken"],
    }

    try:
        # Use global client if available, else fallback (though global should be ready)
        client = GLOBAL_CLIENT
        if not client:
            # Fallback for safety
            async with httpx.AsyncClient(timeout=60.0) as temp_client:
                r = await temp_client.post(TOKEN_URL, headers=_oidc_headers(), json=payload)
                r.raise_for_status()
                data = r.json()
        else:
            r = await client.post(TOKEN_URL, headers=_oidc_headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        new_access = data.get("accessToken")
        new_refresh = data.get("refreshToken", acc.get("refreshToken"))
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "success"
    except httpx.HTTPError as e:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "failed"
        await _db.execute(
            """
            UPDATE accounts
            SET last_refresh_time=?, last_refresh_status=?, updated_at=?
            WHERE id=?
            """,
            (now, status, now, account_id),
        )
        raise HTTPException(status_code=502, detail=f"Token refresh failed: {str(e)}")
    except Exception as e:
        # Ensure last_refresh_time is recorded even on unexpected errors
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "failed"
        await _db.execute(
            """
            UPDATE accounts
            SET last_refresh_time=?, last_refresh_status=?, updated_at=?
            WHERE id=?
            """,
            (now, status, now, account_id),
        )
        raise

    await _db.execute(
        """
        UPDATE accounts
        SET accessToken=?, refreshToken=?, last_refresh_time=?, last_refresh_status=?, updated_at=?
        WHERE id=?
        """,
        (new_access, new_refresh, now, status, now, account_id),
    )

    row2 = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    return _row_to_dict(row2)

async def get_account(account_id: str) -> Dict[str, Any]:
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    return _row_to_dict(row)

async def _update_stats(account_id: str, success: bool, model: str = "unknown") -> None:
    # Update in-memory request stats only (resets on app restart)
    if success:
        REQUEST_STATS.record_success(model, account_id)
    else:
        REQUEST_STATS.record_error(model, account_id)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

async def require_account(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    key = _extract_bearer(authorization) if authorization else x_api_key
    return await resolve_account_for_key(key)

def verify_admin_password(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin password for console access"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized access", "code": "UNAUTHORIZED"}
        )

    password = authorization[7:]  # Remove "Bearer " prefix

    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid password", "code": "INVALID_PASSWORD"}
        )

    return True

# ------------------------------------------------------------------------------
# OpenAI-compatible Chat endpoint
# ------------------------------------------------------------------------------

def _openai_non_streaming_response(
    text: str,
    model: Optional[str],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

def _sse_format(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"

MAX_RETRY_ON_THROTTLE = 3  # Maximum retry attempts when 429 is encountered

@app.post("/v1/messages")
async def claude_messages(request: Request, req: ClaudeRequest, account: Dict[str, Any] = Depends(require_account)):
    """
    Claude-compatible messages endpoint with retry on 429.
    """
    client_ip = _get_client_ip(request)
    print(f"[Request] IP: {client_ip}, Model: {req.model}")
    
    # 1. Convert request
    try:
        aq_request = convert_claude_to_amazonq_request(req)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Request conversion failed: {str(e)}")

    # Helper to send request with a specific account
    async def _send_with_account(acc: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        access = acc.get("accessToken")
        if not access:
            refreshed = await refresh_access_token_in_db(acc["id"])
            access = refreshed.get("accessToken")
            if not access:
                raise HTTPException(status_code=502, detail="Access token unavailable after refresh")
        
        return await send_chat_request(
            access_token=access,
            messages=[],
            model=req.model,
            stream=True,
            client=GLOBAL_CLIENT,
            raw_payload=aq_request
        )

    # 2. Try with retry on 429
    current_account = account
    tried_account_ids: List[str] = []
    event_iter = None
    tracker = None
    last_error = None
    
    for attempt in range(MAX_RETRY_ON_THROTTLE + 1):
        try:
            tried_account_ids.append(current_account["id"])
            _, _, tracker, event_iter = await _send_with_account(current_account)
            
            if not event_iter:
                raise HTTPException(status_code=502, detail="No event stream returned")
            break  # Success, exit retry loop
            
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            last_error = e
            
            if status_code == 429:
                # Throttle this account
                await _throttle_account(current_account["id"])
                print(f"[Retry] Account {current_account['id'][:8]}... got 429, attempt {attempt + 1}/{MAX_RETRY_ON_THROTTLE + 1}")
                
                if attempt < MAX_RETRY_ON_THROTTLE:
                    # Try to get another account
                    candidates = await _list_enabled_accounts_excluding(tried_account_ids, limit=5)
                    if candidates:
                        current_account = random.choice(candidates)
                        print(f"[Retry] Switching to account {current_account['id'][:8]}...")
                        continue
                    else:
                        print("[Retry] No more available accounts")
                
                # All retries exhausted or no more accounts
                raise HTTPException(status_code=429, detail="All accounts rate limited: Too many requests")
            
            elif status_code == 400:
                await _update_stats(current_account["id"], False, map_model_name(req.model))
                raise HTTPException(status_code=400, detail=f"Upstream bad request: {str(e)}")
            else:
                await _update_stats(current_account["id"], False, map_model_name(req.model))
                raise HTTPException(status_code=502, detail=f"Upstream error {status_code}: {str(e)}")
                
        except Exception as e:
            await _update_stats(current_account["id"], False, map_model_name(req.model))
            raise
    
    # 3. Process the stream
    first_event_received = False
    try:
        # Calculate input tokens
        text_to_count = ""
        if req.system:
            if isinstance(req.system, str):
                text_to_count += req.system
            elif isinstance(req.system, list):
                for item in req.system:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_to_count += item.get("text", "")
        
        for msg in req.messages:
            if isinstance(msg.content, str):
                text_to_count += msg.content
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_to_count += item.get("text", "")

        input_tokens = count_tokens(text_to_count, apply_multiplier=True)
        handler = ClaudeStreamHandler(model=req.model, input_tokens=input_tokens)

        # Try to get the first event to ensure the connection is valid
        # This allows us to return proper HTTP error codes before starting the stream
        first_event = None
        try:
            first_event = await event_iter.__anext__()
            first_event_received = True
        except StopAsyncIteration:
            raise HTTPException(status_code=502, detail="Empty response from upstream")
        except Exception as e:
            # If we get an error before the first event, we can still return proper status code
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

        async def event_generator():
            try:
                # Process the first event we already fetched
                if first_event:
                    event_type, payload = first_event
                    async for sse in handler.handle_event(event_type, payload):
                        yield sse
                
                # Process remaining events
                async for event_type, payload in event_iter:
                    async for sse in handler.handle_event(event_type, payload):
                        yield sse
                async for sse in handler.finish():
                    yield sse
                await _update_stats(current_account["id"], True, map_model_name(req.model))
                # Log successful request if enabled (upstream request/response info)
                if REQUEST_LOGGING_ENABLED:
                    log_text = (
                        f"[成功请求] Model: {map_model_name(req.model)}, IP: {client_ip}\n\n"
                        f"--- 上游请求URL ---\n{tracker.upstream_url}\n\n"
                        f"--- 上游请求体 ---\n{tracker.upstream_request_body}\n\n"
                        f"--- 上游响应状态 ---\n{tracker.upstream_status}\n\n"
                        f"--- 上游响应体(首个) ---\n{tracker.upstream_first_event or '(无)'}"
                    )
                    add_structured_log(log_text, kind="request")
            except GeneratorExit:
                # Client disconnected - update stats but don't re-raise
                await _update_stats(current_account["id"], tracker.has_content if tracker else False, map_model_name(req.model))
            except Exception:
                await _update_stats(current_account["id"], False, map_model_name(req.model))
                raise

        if req.stream:
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            # Accumulate for non-streaming
            # This is a bit complex because we need to reconstruct the full response object
            # For now, let's just support streaming as it's the main use case for Claude Code
            # But to be nice, let's try to support non-streaming by consuming the generator
            
            content_blocks = []
            usage = {"input_tokens": 0, "output_tokens": 0}
            stop_reason = None
            
            # We need to parse the SSE strings back to objects... inefficient but works
            # Or we could refactor handler to yield objects.
            # For now, let's just raise error for non-streaming or implement basic text
            # Claude Code uses streaming.
            
            # Let's implement a basic accumulator from the SSE stream
            final_content = []
            
            async for sse_chunk in event_generator():
                data_str = None
                # Each chunk from the generator can have multiple lines ('event:', 'data:').
                # We need to find the 'data:' line.
                for line in sse_chunk.strip().split('\n'):
                    if line.startswith("data:"):
                        data_str = line[6:].strip()
                        break
                
                if not data_str or data_str == "[DONE]":
                    continue
                
                try:
                    data = json.loads(data_str)
                    dtype = data.get("type")
                    
                    if dtype == "content_block_start":
                        idx = data.get("index", 0)
                        while len(final_content) <= idx:
                            final_content.append(None)
                        final_content[idx] = data.get("content_block")
                    
                    elif dtype == "content_block_delta":
                        idx = data.get("index", 0)
                        delta = data.get("delta", {})
                        if final_content[idx]:
                            if delta.get("type") == "text_delta":
                                final_content[idx]["text"] += delta.get("text", "")
                            elif delta.get("type") == "thinking_delta":
                                final_content[idx].setdefault("thinking", "")
                                final_content[idx]["thinking"] += delta.get("thinking", "")
                            elif delta.get("type") == "input_json_delta":
                                if "partial_json" not in final_content[idx]:
                                    final_content[idx]["partial_json"] = ""
                                final_content[idx]["partial_json"] += delta.get("partial_json", "")
                    
                    elif dtype == "content_block_stop":
                        idx = data.get("index", 0)
                        if final_content[idx] and final_content[idx].get("type") == "tool_use":
                            if "partial_json" in final_content[idx]:
                                try:
                                    final_content[idx]["input"] = json.loads(final_content[idx]["partial_json"])
                                except json.JSONDecodeError:
                                    # Keep partial if invalid
                                    final_content[idx]["input"] = {"error": "invalid json", "partial": final_content[idx]["partial_json"]}
                                del final_content[idx]["partial_json"]
                    
                    elif dtype == "message_delta":
                        usage = data.get("usage", usage)
                        stop_reason = data.get("delta", {}).get("stop_reason")
                
                except json.JSONDecodeError:
                    # Ignore lines that are not valid JSON
                    pass
                except Exception:
                    # Broad exception to prevent accumulator from crashing on one bad event
                    traceback.print_exc()
                    pass

            # Final assembly
            final_content_cleaned = []
            for c in final_content:
                if c is not None:
                    # Remove internal state like 'partial_json' before returning
                    c.pop("partial_json", None)
                    final_content_cleaned.append(c)

            return JSONResponse(content={
                "id": f"msg_{uuid.uuid4()}",
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": final_content_cleaned,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": usage
            })

    except httpx.HTTPStatusError as e:
        try:
            if event_iter and hasattr(event_iter, "aclose"):
                await event_iter.aclose()
        except Exception:
            pass
        await _update_stats(current_account["id"], False, map_model_name(req.model))
        status_code = e.response.status_code
        if status_code == 429:
            await _throttle_account(current_account["id"])
            raise HTTPException(status_code=429, detail="Upstream rate limited: Too many requests")
        elif status_code == 400:
            raise HTTPException(status_code=400, detail=f"Upstream bad request: {str(e)}")
        else:
            raise HTTPException(status_code=502, detail=f"Upstream error {status_code}: {str(e)}")
    except Exception as e:
        # Ensure event_iter (if created) is closed to release upstream connection
        try:
            if event_iter and hasattr(event_iter, "aclose"):
                await event_iter.aclose()
        except Exception:
            pass
        await _update_stats(current_account["id"], False, map_model_name(req.model))
        raise

@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(req: ClaudeRequest):
    """
    Count tokens in a message without sending it.
    Compatible with Claude API's /v1/messages/count_tokens endpoint.
    Uses tiktoken for local token counting.
    """
    text_to_count = ""
    
    # Count system prompt tokens
    if req.system:
        if isinstance(req.system, str):
            text_to_count += req.system
        elif isinstance(req.system, list):
            for item in req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count message tokens
    for msg in req.messages:
        if isinstance(msg.content, str):
            text_to_count += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count tool definition tokens if present
    if req.tools:
        text_to_count += json.dumps([tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in req.tools], ensure_ascii=False)
    
    input_tokens = count_tokens(text_to_count, apply_multiplier=True)
    
    return {"input_tokens": input_tokens}

@app.get("/v1/models")
async def list_models(account: Dict[str, Any] = Depends(require_account)):
    data = [{"id": mid, "object": "model", "owned_by": "anthropic"} for mid in SUPPORTED_MODELS]
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, req: ChatCompletionRequest, account: Dict[str, Any] = Depends(require_account)):
    """
    OpenAI-compatible chat endpoint.
    - stream default False
    - messages will be converted into "{role}:\n{content}" and injected into template
    - account is chosen randomly among enabled accounts (API key is for authorization only)
    """
    client_ip = _get_client_ip(request)
    print(f"[Request] IP: {client_ip}, Model: {req.model}")
    
    model = req.model
    do_stream = bool(req.stream)

    async def _send_upstream(stream: bool) -> Tuple[Optional[str], Optional[AsyncGenerator[str, None]], Any]:
        access = account.get("accessToken")
        if not access:
            refreshed = await refresh_access_token_in_db(account["id"])
            access = refreshed.get("accessToken")
            if not access:
                raise HTTPException(status_code=502, detail="Access token unavailable after refresh")
        # Note: send_chat_request signature changed, but we use keyword args so it should be fine if we don't pass raw_payload
        # But wait, the return signature changed too! It now returns 4 values.
        # We need to unpack 4 values.
        result = await send_chat_request(access, [m.model_dump() for m in req.messages], model=model, stream=stream, client=GLOBAL_CLIENT)
        return result[0], result[1], result[2] # Ignore the 4th value (event_stream) for OpenAI endpoint

    if not do_stream:
        try:
            # Calculate prompt tokens
            prompt_text = "".join([m.content for m in req.messages if isinstance(m.content, str)])
            prompt_tokens = count_tokens(prompt_text)

            text, _, tracker = await _send_upstream(stream=False)
            await _update_stats(account["id"], bool(text), map_model_name(model))
            
            completion_tokens = count_tokens(text or "")
            
            response_obj = _openai_non_streaming_response(
                text or "",
                model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # Log successful request if enabled (upstream request/response info)
            if REQUEST_LOGGING_ENABLED:
                log_text = (
                    f"[成功请求] Model: {map_model_name(model)}, IP: {client_ip}\n\n"
                    f"--- 上游请求URL ---\n{tracker.upstream_url}\n\n"
                    f"--- 上游请求体 ---\n{tracker.upstream_request_body}\n\n"
                    f"--- 上游响应状态 ---\n{tracker.upstream_status}\n\n"
                    f"--- 上游响应体(首个) ---\n{tracker.upstream_first_event or '(无)'}"
                )
                add_structured_log(log_text, kind="request")
            
            return JSONResponse(content=response_obj)
        except Exception as e:
            await _update_stats(account["id"], False, map_model_name(model))
            raise
    else:
        created = int(time.time())
        stream_id = f"chatcmpl-{uuid.uuid4()}"
        model_used = model or "unknown"
        
        it = None
        try:
            # Calculate prompt tokens
            prompt_text = "".join([m.content for m in req.messages if isinstance(m.content, str)])
            prompt_tokens = count_tokens(prompt_text)

            _, it, tracker = await _send_upstream(stream=True)
            assert it is not None
            
            async def event_gen() -> AsyncGenerator[str, None]:
                completion_text = ""
                try:
                    # Send role first
                    first_chunk = _sse_format({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_used,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    })
                    yield first_chunk
                    
                    # Stream content
                    async for piece in it:
                        if piece:
                            completion_text += piece
                            chunk = _sse_format({
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_used,
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                            })
                            yield chunk
                    
                    # Send stop and usage
                    completion_tokens = count_tokens(completion_text)
                    stop_chunk = _sse_format({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_used,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                    })
                    yield stop_chunk
                    
                    yield "data: [DONE]\n\n"
                    await _update_stats(account["id"], True, map_model_name(model))
                    # Log successful request if enabled (upstream request/response info)
                    if REQUEST_LOGGING_ENABLED:
                        log_text = (
                            f"[成功请求] Model: {map_model_name(model)}, IP: {client_ip}\n\n"
                            f"--- 上游请求URL ---\n{tracker.upstream_url}\n\n"
                            f"--- 上游请求体 ---\n{tracker.upstream_request_body}\n\n"
                            f"--- 上游响应状态 ---\n{tracker.upstream_status}\n\n"
                            f"--- 上游响应体(首个) ---\n{tracker.upstream_first_event or '(无)'}"
                        )
                        add_structured_log(log_text, kind="request")
                except GeneratorExit:
                    # Client disconnected - update stats but don't re-raise
                    await _update_stats(account["id"], tracker.has_content if tracker else False, map_model_name(model))
                except Exception:
                    await _update_stats(account["id"], tracker.has_content if tracker else False, map_model_name(model))
                    raise
            
            return StreamingResponse(event_gen(), media_type="text/event-stream")
        except Exception as e:
            # Ensure iterator (if created) is closed to release upstream connection
            try:
                if it and hasattr(it, "aclose"):
                    await it.aclose()
            except Exception:
                pass
            await _update_stats(account["id"], False, map_model_name(model))
            raise

# ------------------------------------------------------------------------------
# Device Authorization (URL Login, 5-minute timeout)
# ------------------------------------------------------------------------------

# Dynamic import of auth_flow.py (device-code login helpers)
def _load_auth_flow_module():
    mod_path = BASE_DIR / "auth_flow.py"
    spec = importlib.util.spec_from_file_location("v2_auth_flow", str(mod_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_auth_flow = _load_auth_flow_module()
register_client_min = _auth_flow.register_client_min
device_authorize = _auth_flow.device_authorize
poll_token_device_code = _auth_flow.poll_token_device_code

# In-memory auth sessions (ephemeral)
AUTH_SESSIONS: Dict[str, Dict[str, Any]] = {}

class AuthStartBody(BaseModel):
    label: Optional[str] = None
    enabled: Optional[bool] = True

class AdminLoginRequest(BaseModel):
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    message: str

async def _create_account_from_tokens(
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: Optional[str],
    label: Optional[str],
    enabled: bool,
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    acc_id = str(uuid.uuid4())
    await _db.execute(
        """
        INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            acc_id,
            label,
            client_id,
            client_secret,
            refresh_token,
            access_token,
            None,
            now,
            "success",
            now,
            now,
            1 if enabled else 0,
        ),
    )
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (acc_id,))
    return _row_to_dict(row)

# 管理控制台相关端点 - 仅在启用时注册
if CONSOLE_ENABLED:
    # ------------------------------------------------------------------------------
    # Admin Authentication Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/api/login", response_model=AdminLoginResponse)
    async def admin_login(request: AdminLoginRequest) -> AdminLoginResponse:
        """Admin login endpoint - password only"""
        if request.password == ADMIN_PASSWORD:
            return AdminLoginResponse(
                success=True,
                message="Login successful"
            )
        else:
            return AdminLoginResponse(
                success=False,
                message="Invalid password"
            )

    @app.get("/v2/gitstore/status")
    async def gitstore_status(_: bool = Depends(verify_admin_password)):
        if not GITSTORE_SYNC:
            return {"enabled": False, "mode": "LOCAL", "pending": False, "error": None}
        st = GITSTORE_SYNC.status()
        return {"enabled": True, "mode": st.mode, "pending": st.pending, "error": st.error}

    @app.post("/v2/gitstore/reconnect")
    async def gitstore_reconnect(_: bool = Depends(verify_admin_password)):
        """Manually attempt to reconnect gitstore. Call this when clicking the status badge."""
        if not GITSTORE_SYNC:
            return {"success": False, "error": "gitstore not configured"}
        success = await GITSTORE_SYNC.reconnect()
        st = GITSTORE_SYNC.status()
        return {"success": success, "mode": st.mode, "pending": st.pending, "error": st.error}

    @app.get("/v2/meta/storage")
    async def meta_storage(_: bool = Depends(verify_admin_password)):
        backend = _db_backend_name()
        if backend != "sqlite":
            return {"backend": backend, "gitstore": {"enabled": False, "mode": None, "pending": False, "error": None}}
        if not GITSTORE_SYNC:
            return {"backend": "sqlite", "gitstore": {"enabled": False, "mode": "LOCAL", "pending": False, "error": None}}
        st = GITSTORE_SYNC.status()
        return {"backend": "sqlite", "gitstore": {"enabled": True, "mode": st.mode, "pending": st.pending, "error": st.error}}

    @app.get("/v2/meta/stats")
    async def meta_stats(_: bool = Depends(verify_admin_password)):
        """Return in-memory request statistics (total, success, error)."""
        return REQUEST_STATS.get()

    @app.post("/v2/meta/logs/toggle")
    async def meta_logs_toggle(body: Dict[str, Any], _: bool = Depends(verify_admin_password)):
        enabled = bool(body.get("enabled", False))
        if enabled:
            enable_capture()
        else:
            disable_capture()
        return {"enabled": capture_status()}

    @app.post("/v2/meta/logs/requests/toggle")
    async def meta_logs_requests_toggle(body: Dict[str, Any], _: bool = Depends(verify_admin_password)):
        """Toggle request logging (successful requests with request/response snippets)."""
        global REQUEST_LOGGING_ENABLED
        enabled = bool(body.get("enabled", False))
        REQUEST_LOGGING_ENABLED = enabled
        return {"enabled": REQUEST_LOGGING_ENABLED}

    @app.get("/v2/meta/logs")
    async def meta_logs(after: int = 0, limit: int = 200, _: bool = Depends(verify_admin_password)):
        limit = max(1, min(limit or 200, 1000))
        entries = get_logs(after=after, limit=limit)
        return {
            "enabled": capture_status(),
            "request_logging_enabled": REQUEST_LOGGING_ENABLED,
            "logs": [
                {"seq": e.seq, "ts": e.ts, "text": e.text, "kind": e.kind}
                for e in entries
            ],
        }

    @app.get("/v2/meta/egress_ip")
    async def meta_egress_ip(_: bool = Depends(verify_admin_password)):
        """Return egress IP with best-effort geo; always return IP if obtainable."""
        client = GLOBAL_CLIENT
        ip = None
        country = None
        city = None
        geo_pending = False

        async def _ipify(c):
            r = await c.get("https://api.ipify.org", params={"format": "json"}, timeout=3.0)
            r.raise_for_status()
            return r.json().get("ip") or r.text.strip()

        async def _ipapi(c):
            r = await c.get("https://ipapi.co/json/?lang=zh-CN", timeout=3.0)
            r.raise_for_status()
            d = r.json()
            return (
                d.get("ip") or d.get("query"),
                d.get("country_name"),
                d.get("city"),
            )

        try:
            if client is None:
                async with httpx.AsyncClient(timeout=5.0) as temp:
                    try:
                        ip = await _ipify(temp)
                    except Exception:
                        pass
                    try:
                        ip2, country, city = await _ipapi(temp)
                        if ip2:
                            ip = ip or ip2
                    except Exception:
                        geo_pending = True if ip else False
            else:
                try:
                    ip = await _ipify(client)
                except Exception:
                    pass
                try:
                    ip2, country, city = await _ipapi(client)
                    if ip2:
                        ip = ip or ip2
                except Exception:
                    geo_pending = True if ip else False

            if not ip:
                raise RuntimeError("unable to determine egress IP")

            return {"ip": ip, "country": country, "city": city, "geo_pending": geo_pending}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Egress IP fetch failed: {e}")

    @app.get("/login", response_class=FileResponse)
    def login_page():
        """Serve the login page"""
        path = BASE_DIR / "frontend" / "login.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/login.html not found")
        return FileResponse(str(path))

    # ------------------------------------------------------------------------------
    # Device Authorization Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/v2/auth/start")
    async def auth_start(body: AuthStartBody, _: bool = Depends(verify_admin_password)):
        """
        Start device authorization and return verification URL for user login.
        Session lifetime capped at 5 minutes on claim.
        """
        try:
            cid, csec = await register_client_min()
            dev = await device_authorize(cid, csec)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

        auth_id = str(uuid.uuid4())
        sess = {
            "clientId": cid,
            "clientSecret": csec,
            "deviceCode": dev.get("deviceCode"),
            "interval": int(dev.get("interval", 1)),
            "expiresIn": int(dev.get("expiresIn", 600)),
            "verificationUriComplete": dev.get("verificationUriComplete"),
            "userCode": dev.get("userCode"),
            "startTime": int(time.time()),
            "label": body.label,
            "enabled": True if body.enabled is None else bool(body.enabled),
            "status": "pending",
            "error": None,
            "accountId": None,
        }
        AUTH_SESSIONS[auth_id] = sess
        return {
            "authId": auth_id,
            "verificationUriComplete": sess["verificationUriComplete"],
            "userCode": sess["userCode"],
            "expiresIn": sess["expiresIn"],
            "interval": sess["interval"],
        }

    @app.get("/v2/auth/status/{auth_id}")
    async def auth_status(auth_id: str, _: bool = Depends(verify_admin_password)):
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        now_ts = int(time.time())
        deadline = sess["startTime"] + min(int(sess.get("expiresIn", 600)), 300)
        remaining = max(0, deadline - now_ts)
        return {
            "status": sess.get("status"),
            "remaining": remaining,
            "error": sess.get("error"),
            "accountId": sess.get("accountId"),
        }

    @app.post("/v2/auth/claim/{auth_id}")
    async def auth_claim(auth_id: str, _: bool = Depends(verify_admin_password)):
        """
        Block up to 5 minutes to exchange the device code for tokens after user completed login.
        On success, creates an enabled account and returns it.
        """
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        if sess.get("status") in ("completed", "timeout", "error"):
            return {
                "status": sess["status"],
                "accountId": sess.get("accountId"),
                "error": sess.get("error"),
            }
        try:
            toks = await poll_token_device_code(
                sess["clientId"],
                sess["clientSecret"],
                sess["deviceCode"],
                sess["interval"],
                sess["expiresIn"],
                max_timeout_sec=300,  # 5 minutes
            )
            access_token = toks.get("accessToken")
            refresh_token = toks.get("refreshToken")
            if not access_token:
                raise HTTPException(status_code=502, detail="No accessToken returned from OIDC")

            acc = await _create_account_from_tokens(
                sess["clientId"],
                sess["clientSecret"],
                access_token,
                refresh_token,
                sess.get("label"),
                sess.get("enabled", True),
            )
            sess["status"] = "completed"
            sess["accountId"] = acc["id"]
            return {
                "status": "completed",
                "account": acc,
            }
        except TimeoutError:
            sess["status"] = "timeout"
            raise HTTPException(status_code=408, detail="Authorization timeout (5 minutes)")
        except httpx.HTTPError as e:
            sess["status"] = "error"
            sess["error"] = str(e)
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

    # ------------------------------------------------------------------------------
    # Accounts Management API
    # ------------------------------------------------------------------------------

    @app.post("/v2/accounts")
    async def create_account(body: AccountCreate, _: bool = Depends(verify_admin_password)):
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        acc_id = str(uuid.uuid4())
        other_str = json.dumps(body.other, ensure_ascii=False) if body.other is not None else None
        enabled_val = 1 if (body.enabled is None or body.enabled) else 0
        await _db.execute(
            """
            INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                acc_id,
                body.label,
                body.clientId,
                body.clientSecret,
                body.refreshToken,
                body.accessToken,
                other_str,
                None,
                "never",
                now,
                now,
                enabled_val,
            ),
        )
        row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (acc_id,))
        return _row_to_dict(row)


    async def _verify_and_enable_accounts(account_ids: List[str]):
        """后台异步验证并启用账号"""
        for acc_id in account_ids:
            try:
                # 必须先获取完整的账号信息
                account = await get_account(acc_id)
                verify_success, fail_reason = await verify_account(account)
                now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

                if verify_success:
                    await _db.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, acc_id))
                elif fail_reason:
                    other_dict = account.get("other", {}) or {}
                    other_dict['failedReason'] = fail_reason
                    await _db.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, acc_id))
            except Exception as e:
                print(f"Error verifying account {acc_id}: {e}")
                traceback.print_exc()

    @app.post("/v2/accounts/feed")
    async def create_accounts_feed(request: BatchAccountCreate, _: bool = Depends(verify_admin_password)):
        """
        统一的投喂接口，接收账号列表，立即存入并后台异步验证。
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        new_account_ids = []

        for i, account_data in enumerate(request.accounts):
            acc_id = str(uuid.uuid4())
            other_dict = account_data.other or {}
            other_dict['source'] = 'feed'
            other_str = json.dumps(other_dict, ensure_ascii=False)

            await _db.execute(
                """
                INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    acc_id,
                    account_data.label or f"批量账号 {i+1}",
                    account_data.clientId,
                    account_data.clientSecret,
                    account_data.refreshToken,
                    account_data.accessToken,
                    other_str,
                    None,
                    "never",
                    now,
                    now,
                    0,  # 初始为禁用状态
                ),
            )
            new_account_ids.append(acc_id)

        # 启动后台任务进行验证，不阻塞当前请求
        if new_account_ids:
            asyncio.create_task(_verify_and_enable_accounts(new_account_ids))

        return {
            "status": "processing",
            "message": f"{len(new_account_ids)} accounts received and are being verified in the background.",
            "account_ids": new_account_ids
        }

    @app.get("/v2/accounts")
    async def list_accounts(_: bool = Depends(verify_admin_password), enabled: Optional[bool] = None, sort_by: str = "created_at", sort_order: str = "desc"):
        query = "SELECT * FROM accounts"
        params = []
        if enabled is not None:
            query += " WHERE enabled=?"
            params.append(1 if enabled else 0)
        sort_field = "created_at" if sort_by not in ["created_at", "success_count"] else sort_by
        order = "DESC" if sort_order.lower() == "desc" else "ASC"
        query += f" ORDER BY {sort_field} {order}"
        rows = await _db.fetchall(query, tuple(params) if params else ())
        accounts = []
        for r in rows:
            acc = _row_to_dict(r)
            acc_id = acc["id"]
            
            # Override with in-memory stats (resets on app restart)
            mem_stats = REQUEST_STATS.get_account_stats(acc_id)
            acc["success_count"] = mem_stats["success"]
            acc["error_count"] = mem_stats["error"]
            
            # Add throttle info from memory
            throttle_until = THROTTLE_TRACKER.throttled_until.get(acc_id)
            error_count = THROTTLE_TRACKER.error_counts.get(acc_id, 0)
            if throttle_until and time.time() < throttle_until:
                acc["throttle_info"] = {
                    "is_throttled": True,
                    "until": throttle_until,
                    "remaining_seconds": int(throttle_until - time.time()),
                    "error_count": error_count,
                    "is_max_level": error_count >= 5  # 30s * 2^4 = 480s > 300s max, so 5 errors means max
                }
            else:
                acc["throttle_info"] = {"is_throttled": False, "error_count": 0, "is_max_level": False}
            accounts.append(acc)
        return {"accounts": accounts, "count": len(accounts)}

    @app.get("/v2/accounts/{account_id}")
    async def get_account_detail(account_id: str, _: bool = Depends(verify_admin_password)):
        return await get_account(account_id)

    @app.delete("/v2/accounts/{account_id}")
    async def delete_account(account_id: str, _: bool = Depends(verify_admin_password)):
        rowcount = await _db.execute("DELETE FROM accounts WHERE id=?", (account_id,))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        return {"deleted": account_id}

    @app.patch("/v2/accounts/{account_id}")
    async def update_account(account_id: str, body: AccountUpdate, _: bool = Depends(verify_admin_password)):
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        fields = []
        values: List[Any] = []

        if body.label is not None:
            fields.append("label=?"); values.append(body.label)
        if body.clientId is not None:
            fields.append("clientId=?"); values.append(body.clientId)
        if body.clientSecret is not None:
            fields.append("clientSecret=?"); values.append(body.clientSecret)
        if body.refreshToken is not None:
            fields.append("refreshToken=?"); values.append(body.refreshToken)
        if body.accessToken is not None:
            fields.append("accessToken=?"); values.append(body.accessToken)
        if body.other is not None:
            fields.append("other=?"); values.append(json.dumps(body.other, ensure_ascii=False))
        if body.enabled is not None:
            fields.append("enabled=?"); values.append(1 if body.enabled else 0)

        if not fields:
            return await get_account(account_id)

        fields.append("updated_at=?"); values.append(now)
        values.append(account_id)

        rowcount = await _db.execute(f"UPDATE accounts SET {', '.join(fields)} WHERE id=?", tuple(values))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
        return _row_to_dict(row)

    @app.post("/v2/accounts/{account_id}/refresh")
    async def manual_refresh(account_id: str, _: bool = Depends(verify_admin_password)):
        return await refresh_access_token_in_db(account_id)

    # ------------------------------------------------------------------------------
    # Simple Frontend (minimal dev test page; full UI in v2/frontend/index.html)
    # ------------------------------------------------------------------------------

    # Frontend inline HTML removed; serving ./frontend/index.html instead (see route below)
    # Note: This route is NOT protected - the HTML file is served freely,
    # but the frontend JavaScript checks authentication and redirects to /login if needed.
    # All API endpoints remain protected.

    @app.get("/", response_class=FileResponse)
    def index():
        path = BASE_DIR / "frontend" / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(str(path))

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# ------------------------------------------------------------------------------
# Startup / Shutdown Events
# ------------------------------------------------------------------------------

# async def _verify_disabled_accounts_loop():
#     """后台验证禁用账号任务"""
#     while True:
#         try:
#             await asyncio.sleep(1800)
#             async with _conn() as conn:
#                 accounts = await _list_disabled_accounts(conn)
#                 if accounts:
#                     for account in accounts:
#                         other = account.get('other')
#                         if other:
#                             try:
#                                 other_dict = json.loads(other) if isinstance(other, str) else other
#                                 if other_dict.get('failedReason') == 'AccessDenied':
#                                     continue
#                             except:
#                                 pass
#                         try:
#                             verify_success, fail_reason = await verify_account(account)
#                             now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
#                             if verify_success:
#                                 await conn.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, account['id']))
#                             elif fail_reason:
#                                 other_dict = {}
#                                 if account.get('other'):
#                                     try:
#                                         other_dict = json.loads(account['other']) if isinstance(account['other'], str) else account['other']
#                                     except:
#                                         pass
#                                 other_dict['failedReason'] = fail_reason
#                                 await conn.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, account['id']))
#                             await conn.commit()
#                         except Exception:
#                             pass
#         except Exception:
#             pass

@app.on_event("startup")
async def startup_event():
    """Initialize database and start background tasks on startup."""
    await _init_global_client()
    # GitStore 必须在 _ensure_db() 之前准备，以便从远端恢复数据
    if GITSTORE_SYNC:
        try:
            print("[GitStore] 正在准备 GitStore...")
            await GITSTORE_SYNC.prepare()
            status = GITSTORE_SYNC.status()
            print(f"[GitStore] 准备完成，模式: {status.mode}, 错误: {status.error}")
            if status.mode != "LOCAL":
                GITSTORE_SYNC.start_background()
                print("[GitStore] 后台同步任务已启动")
        except Exception:
            traceback.print_exc()
    await _ensure_db()
    asyncio.create_task(_refresh_stale_tokens())
    # asyncio.create_task(_verify_disabled_accounts_loop())

@app.on_event("shutdown")
async def shutdown_event():
    await _close_global_client()
    if GITSTORE_SYNC:
        try:
            # 关闭前做最后一次快照推送，避免丢失最近的变更
            print("[GitStore] 正在执行关闭前快照推送...")
            await GITSTORE_SYNC.snapshot_and_push()
            print("[GitStore] 快照推送完成")
            await GITSTORE_SYNC.stop_background()
        except Exception:
            traceback.print_exc()
    await close_db()
