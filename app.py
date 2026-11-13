"""
ReturnGuard FastAPI Entrypoint
Forwards application instance from main.py for backwards compatibility.
"""
from main import app, SinglePredictionInput, FEATURES

__all__ = ["app", "SinglePredictionInput", "FEATURES"]

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
