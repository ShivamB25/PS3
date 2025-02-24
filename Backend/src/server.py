"""FastAPI server for price optimization system."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from price_optimizer.api.price_api import router as price_router
from price_optimizer.api.visualization_endpoints import router as viz_router
from price_optimizer.api.network_endpoints import router as network_router

app = FastAPI(title="Price Optimization API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(price_router, prefix="/api", tags=["price"])
app.include_router(viz_router, prefix="/api/visualize", tags=["visualization"])
app.include_router(network_router, prefix="/network", tags=["network"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
