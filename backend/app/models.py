from sqlalchemy import Column, Integer, String, DateTime
from .database import Base
# define how data is stored in the database.
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    fullname = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    reset_token = Column(String, nullable=True)
    token_expiry = Column(DateTime, nullable=True)
