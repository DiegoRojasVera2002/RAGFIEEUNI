from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_service import RAGService
from src.utils import PerformanceTracker, validate_query
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG UNI FIEE - Refactorizado", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios
try:
    rag_service = RAGService()
    performance_tracker = PerformanceTracker()
    logger.info("✅ Servicios inicializados correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando servicios: {e}")
    rag_service = None

class QueryRequest(BaseModel):
    message: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    response: str
    intent: str
    sources_count: int
    processing_time: float
    success: bool

@app.get("/")
async def root():
    return {"message": "RAG UNI FIEE - Sistema Refactorizado", "version": "2.0.0"}

@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not rag_service:
        raise HTTPException(status_code=500, detail="Servicio no disponible")
    
    if not validate_query(request.message):
        raise HTTPException(status_code=400, detail="Consulta inválida")
    
    try:
        result = rag_service.process_query(request.message)
        
        # Registrar métricas
        performance_tracker.track_query(
            result['processing_time'], 
            result['success']
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    stats = performance_tracker.get_stats()
    return {
        "status": "healthy" if rag_service else "error",
        "metrics": stats,
        "cache_size": len(rag_service.cache) if rag_service else 0
    }

@app.get("/api/stats")
async def get_stats():
    if not rag_service:
        return {"error": "Servicio no disponible"}
    
    return {
        "performance": performance_tracker.get_stats(),
        "cache_entries": len(rag_service.cache)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)