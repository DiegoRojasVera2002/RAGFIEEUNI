import time
from datetime import datetime

class PerformanceTracker:
    def __init__(self):
        self.metrics = []
    
    def track_query(self, processing_time: float, success: bool):
        """Registrar métrica de consulta"""
        self.metrics.append({
            'timestamp': datetime.now(),
            'processing_time': processing_time,
            'success': success
        })
        
        # Mantener solo últimos 100 registros
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]
    
    def get_stats(self) -> dict:
        """Obtener estadísticas"""
        if not self.metrics:
            return {"avg_time": 0, "success_rate": 0, "total_queries": 0}
        
        avg_time = sum(m['processing_time'] for m in self.metrics) / len(self.metrics)
        success_rate = sum(1 for m in self.metrics if m['success']) / len(self.metrics)
        
        return {
            "avg_time": round(avg_time, 2),
            "success_rate": round(success_rate, 2),
            "total_queries": len(self.metrics)
        }

def validate_query(query: str) -> bool:
    """Validar consulta de entrada"""
    return bool(query and len(query.strip()) > 0 and len(query) <= 1000)