"""
Database abstraction layer supporting SQLite, PostgreSQL, and MySQL.
Backend selection is based on DATABASE_URL environment variable:
- postgres://... or postgresql://... -> PostgreSQL
- mysql://... -> MySQL
- Not set -> SQLite (default)
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

import aiosqlite

# Optional imports for other backends
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import aiomysql
    HAS_AIOMYSQL = True
except ImportError:
    HAS_AIOMYSQL = False


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connection and ensure schema exists."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connections."""
        pass

    @abstractmethod
    async def execute(self, query: str, params: tuple = ()) -> int:
        """Execute a query and return affected row count."""
        pass

    @abstractmethod
    async def fetchone(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dict."""
        pass

    @abstractmethod
    async def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows as list of dicts."""
        pass


class SQLiteBackend(DatabaseBackend):
    """SQLite database backend using aiosqlite."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._initialized = False
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        
        # Performance tuning PRAGMAs
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous = NORMAL;")
        await self._conn.execute("PRAGMA cache_size = -65536; -- 64MB")
        await self._conn.execute("PRAGMA temp_store = MEMORY;")
        
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                label TEXT,
                clientId TEXT,
                clientSecret TEXT,
                refreshToken TEXT,
                accessToken TEXT,
                other TEXT,
                last_refresh_time TEXT,
                last_refresh_status TEXT,
                created_at TEXT,
                updated_at TEXT,
                enabled INTEGER DEFAULT 1,
                error_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                throttled_until REAL DEFAULT 0
            )
        """)
        
        # Migration: add throttled_until if not exists
        try:
            await self._conn.execute("ALTER TABLE accounts ADD COLUMN throttled_until REAL DEFAULT 0")
            await self._conn.commit()
        except Exception:
            pass
        
        # Create indexes for performance
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_accounts_enabled ON accounts (enabled);")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_accounts_created_at ON accounts (created_at);")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_accounts_success_count ON accounts (success_count);")

        await self._conn.commit()
        self._initialized = True

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
        self._initialized = False

    async def execute(self, query: str, params: tuple = ()) -> int:
        cursor = await self._conn.execute(query, params)
        await self._conn.commit()
        return cursor.rowcount

    async def fetchone(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        self._conn.row_factory = aiosqlite.Row
        async with self._conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        self._conn.row_factory = aiosqlite.Row
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


class PostgresBackend(DatabaseBackend):
    """PostgreSQL database backend using asyncpg."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: "Optional[asyncpg.pool.Pool]" = None
        self._initialized = False

    async def initialize(self) -> None:
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg is required for PostgreSQL support. Install with: pip install asyncpg")

        self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=20)

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    clientId TEXT,
                    clientSecret TEXT,
                    refreshToken TEXT,
                    accessToken TEXT,
                    other TEXT,
                    last_refresh_time TEXT,
                    last_refresh_status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    enabled INTEGER DEFAULT 1,
                    error_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    throttled_until REAL DEFAULT 0
                )
            """)
            # Migration: add throttled_until if not exists
            try:
                await conn.execute("ALTER TABLE accounts ADD COLUMN throttled_until REAL DEFAULT 0")
            except Exception:
                pass
        self._initialized = True

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False

    def _convert_placeholders(self, query: str) -> str:
        """Convert ? placeholders to $1, $2, etc."""
        result = []
        param_num = 0
        i = 0
        while i < len(query):
            if query[i] == '?':
                param_num += 1
                result.append(f'${param_num}')
            else:
                result.append(query[i])
            i += 1
        return ''.join(result)

    async def execute(self, query: str, params: tuple = ()) -> int:
        pg_query = self._convert_placeholders(query)
        async with self._pool.acquire() as conn:
            result = await conn.execute(pg_query, *params)
            # asyncpg returns string like "UPDATE 1"
            try:
                return int(result.split()[-1])
            except (ValueError, IndexError):
                return 0

    async def fetchone(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        pg_query = self._convert_placeholders(query)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(pg_query, *params)
            return dict(row) if row else None

    async def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        pg_query = self._convert_placeholders(query)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(pg_query, *params)
            return [dict(row) for row in rows]


class MySQLBackend(DatabaseBackend):
    """MySQL database backend using aiomysql."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None
        self._initialized = False
        self._config = self._parse_dsn(dsn)

    def _parse_dsn(self, dsn: str) -> Dict[str, Any]:
        """Parse MySQL DSN into connection parameters."""
        # mysql://user:password@host:port/database
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(dsn)
        config = {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 3306,
            'user': parsed.username or 'root',
            'password': parsed.password or '',
            'db': parsed.path.lstrip('/') if parsed.path else 'test',
        }
        # Handle SSL
        query = parse_qs(parsed.query)
        if 'ssl' in query or 'sslmode' in query or 'ssl-mode' in query:
            config['ssl'] = True
        return config

    async def initialize(self) -> None:
        if not HAS_AIOMYSQL:
            raise ImportError("aiomysql is required for MySQL support. Install with: pip install aiomysql")

        self._pool = await aiomysql.create_pool(
            host=self._config['host'],
            port=self._config['port'],
            user=self._config['user'],
            password=self._config['password'],
            db=self._config['db'],
            minsize=1,
            maxsize=20,
            autocommit=True
        )

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS accounts (
                        id VARCHAR(255) PRIMARY KEY,
                        label TEXT,
                        clientId TEXT,
                        clientSecret TEXT,
                        refreshToken TEXT,
                        accessToken TEXT,
                        other TEXT,
                        last_refresh_time TEXT,
                        last_refresh_status TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        enabled INT DEFAULT 1,
                        error_count INT DEFAULT 0,
                        success_count INT DEFAULT 0,
                        throttled_until DOUBLE DEFAULT 0
                    )
                """)
                # Migration: add throttled_until if not exists
                try:
                    await cur.execute("ALTER TABLE accounts ADD COLUMN throttled_until DOUBLE DEFAULT 0")
                except Exception:
                    pass
        self._initialized = True

    async def close(self) -> None:
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            self._initialized = False

    def _convert_placeholders(self, query: str) -> str:
        """Convert ? placeholders to %s for MySQL."""
        return query.replace('?', '%s')

    async def execute(self, query: str, params: tuple = ()) -> int:
        mysql_query = self._convert_placeholders(query)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(mysql_query, params)
                return cur.rowcount

    async def fetchone(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        mysql_query = self._convert_placeholders(query)
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(mysql_query, params)
                return await cur.fetchone()

    async def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        mysql_query = self._convert_placeholders(query)
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(mysql_query, params)
                return await cur.fetchall()


# Global database instance
_db: Optional[DatabaseBackend] = None


def get_database_backend() -> DatabaseBackend:
    """Get the configured database backend based on DATABASE_URL."""
    global _db
    if _db is not None:
        return _db

    database_url = os.getenv('DATABASE_URL', '').strip()

    if database_url.startswith(('postgres://', 'postgresql://')):
        # Fix common postgres:// to postgresql:// for asyncpg
        dsn = database_url.replace('postgres://', 'postgresql://', 1) if database_url.startswith('postgres://') else database_url
        _db = PostgresBackend(dsn)
        print(f"[DB] Using PostgreSQL backend")
    elif database_url.startswith('mysql://'):
        _db = MySQLBackend(database_url)
        print(f"[DB] Using MySQL backend")
    else:
        # Default to SQLite
        base_dir = Path(__file__).resolve().parent
        db_path = base_dir / "data.sqlite3"
        _db = SQLiteBackend(db_path)
        print(f"[DB] Using SQLite backend: {db_path}")

    return _db


async def init_db() -> DatabaseBackend:
    """Initialize and return the database backend."""
    db = get_database_backend()
    await db.initialize()
    return db


async def close_db() -> None:
    """Close the database backend."""
    global _db
    if _db:
        await _db.close()
        _db = None


# Helper functions for common operations
def row_to_dict(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a database row to dict with JSON parsing for 'other' field."""
    if row is None:
        return None
    d = dict(row)
    if d.get("other"):
        try:
            d["other"] = json.loads(d["other"])
        except Exception:
            pass
    # normalize enabled to bool
    if "enabled" in d and d["enabled"] is not None:
        try:
            d["enabled"] = bool(int(d["enabled"]))
        except Exception:
            d["enabled"] = bool(d["enabled"])
    return d
