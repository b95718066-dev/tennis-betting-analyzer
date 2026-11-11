from fastapi import FastAPI

app = FastAPI(
    title="Tennis Betting Analyzer",
    description="Advanced tennis analytics and prediction API",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {
        "project": "Tennis Betting Analyzer",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "tennis-betting-analyzer",
    }
