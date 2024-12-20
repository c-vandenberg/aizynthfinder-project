import sqlite3
import logging

def get_unique_count(db_path: str = 'seen_pairs.db') -> int:
    logger = logging.getLogger(__name__)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pairs")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.exception(f"Failed to get unique count from database {db_path}.")
        raise