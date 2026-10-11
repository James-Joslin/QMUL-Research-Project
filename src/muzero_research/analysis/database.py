"""Optional PostgreSQL-to-CSV import; connection details are never embedded in code."""

import os
from pathlib import Path


def import_table(table, output):
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("Set DATABASE_URL to use the optional database import adapter")
    import pandas as pd
    from sqlalchemy import create_engine

    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_table(table, connection)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path
    finally:
        engine.dispose()
