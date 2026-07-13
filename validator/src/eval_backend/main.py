from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import admin_router, router
from .core.config import Settings
from .db import Base, build_engine, build_session_factory, ensure_schema
from .services.admin_auth import seed_admin_user
from .services.runtime_config import seed_runtime_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    settings.ensure_dirs()
    engine = build_engine(settings)
    Base.metadata.create_all(bind=engine)
    ensure_schema(engine)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = build_session_factory(engine)
    with app.state.session_factory() as session:
        seed_admin_user(session, settings)
        seed_runtime_config(session, settings)
        session.commit()
    yield


def create_app() -> FastAPI:
    settings = Settings.load()
    app = FastAPI(title="MiniRouter Validator", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(admin_router)
    return app


app = create_app()
