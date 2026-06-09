from sqlmodel import SQLModel, create_engine, Session
import os

# Use environment-configurable database path
DB_PATH = os.getenv("FOODGUARD_DB_PATH", "/data/foodguard.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Initialize database schema."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get a new database session."""
    return Session(engine)
