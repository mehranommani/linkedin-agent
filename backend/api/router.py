"""
API Router
==========
Aggregates all API routers.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.posts import router as posts_router
from backend.api.agent import router as agent_router
from backend.api.sources import router as sources_router
from backend.api.config_api import router as config_router
from backend.api.evaluations import router as evaluations_router
from backend.api.feedback import router as feedback_router

api_router = APIRouter(prefix="/api")
api_router.include_router(posts_router)
api_router.include_router(agent_router)
api_router.include_router(sources_router)
api_router.include_router(config_router)
api_router.include_router(evaluations_router)
api_router.include_router(feedback_router)
