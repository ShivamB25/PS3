"""Price optimization API package."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .price_api import router as price_router
from .visualization_endpoints import router as viz_router

app = FastAPI(title="Price Optimization API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core price optimization endpoints
app.include_router(price_router)

# Include visualization endpoints
app.include_router(
    viz_router,
    tags=["visualization"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Visualization error"},
    },
)
