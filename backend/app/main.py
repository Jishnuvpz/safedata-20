from fastapi import FastAPI
from app.routers import anonymize
from fastapi import FastAPI
from app.routers import anonymize

app = FastAPI()
app.include_router(anonymize.router)

app = FastAPI(title="Anonify Phase 2")
app.include_router(anonymize.router, prefix="/api", tags=["Anonymize"])
@app.get("/")
async def root():
 return {"message": "Welcome to Anonify Phase 2!"}