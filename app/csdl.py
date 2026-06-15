import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

_url = os.getenv(
    "DATABASE_URL",
    "mysql+aiomysql://{user}:{password}@{host}:{port}/{db}".format(
        user = os.getenv("DB_USER", "root"),
        password = os.getenv("DB_PASSWORD", "password"),
        host = os.getenv("DB_HOST", "localhost"),
        port = os.getenv("DB_PORT", "3306"),
        db = os.getenv("DB_NAME", "sdrone_db"),
    )
)

engine = create_async_engine(
    _url,
    pool_pre_ping = True,
    pool_size = 10,
    max_overflow = 20,
    echo = False,
)

AsyncSessionLocal = async_sessionmaker(
    bind = engine,
    class_ = AsyncSession,
    expire_on_commit= False,
)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise