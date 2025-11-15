#!/usr/bin/env python3
import sqlite3
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data.sqlite3"


def delete_disabled_accounts() -> int:
    """
    Delete all accounts where enabled=0 from the SQLite database.
    Returns the number of rows deleted.
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return 0

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            # Ensure table exists
            tbl_cur = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='accounts'"
            )
            if (tbl_cur.fetchone() or [0])[0] == 0:
                print("Table 'accounts' not found in database.")
                return 0

            # Check column 'enabled' existence
            cols = [row[1] for row in conn.execute("PRAGMA table_info(accounts)").fetchall()]
            if "enabled" not in cols:
                print("Column 'enabled' not found in 'accounts' table.")
                return 0

            # Count first for clear reporting, then delete
            count = (conn.execute("SELECT COUNT(*) FROM accounts WHERE enabled=0").fetchone() or [0])[0]
            conn.execute("DELETE FROM accounts WHERE enabled=0")
            conn.commit()
            print(f"Deleted {count} disabled account(s).")
            return int(count)
    except sqlite3.Error as e:
        print(f"SQLite error: {e}", file=sys.stderr)
        return 0


def main() -> None:
    deleted = delete_disabled_accounts()
    # exit code 0 even if none deleted; non-zero only on sqlite error already handled
    sys.exit(0)


if __name__ == "__main__":
    main()