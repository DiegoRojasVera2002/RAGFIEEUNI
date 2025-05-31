import openai
from openai import OpenAI
from qdrant_client import QdrantClient
import time
import logging
from .config import Config

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        Config.validate()
        
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.llm_client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        self.cache = {}
        
    def classify_intent(self, query: str) -> str:
        """Clasificar intención de la consulta"""
        try:
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['fecha', 'examen', 'cronograma']):
                return 'fechas_examenes'
            elif any(word in query_lower for word in ['matricula', 'inscripcion']):
                return 'proceso_matricula'
            elif any(word in query_lower for word in ['costo', 'precio', 'pago']):
                return 'costos_pagos'
            elif any(word in query_lower for word in ['hola', 'gracias', 'buenos']):
                return 'conversacion_casual'
            else:
                return 'consulta_general'
                
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            return 'consulta_general'
    
    def get_embedding(self, text: str) -> list:
        """Obtener embedding de texto"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=Config.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error obteniendo embedding: {e}")
            return []
    
    def search_documents(self, query_embedding: list, limit: int = 5) -> list:
        """Buscar documentos relevantes"""
        try:
            results = self.qdrant_client.search(
                collection_name=Config.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.3
            )
            return results
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def generate_response(self, query: str, contexts: list) -> str:
        """Generar respuesta usando LLM"""
        try:
            if not contexts:
                return "No encontré información específica sobre tu consulta."
            
            context_text = "\n".join([ctx.payload['text'][:200] for ctx in contexts])
            
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "user",
                    "content": f"Contexto: {context_text}\n\nPregunta: {query}\n\nResponde de manera útil:"
                }],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return "Disculpa, hubo un error procesando tu consulta."
    
    def process_query(self, query: str) -> dict:
        """Procesar consulta completa"""
        start_time = time.time()
        
        # Verificar cache
        if query in self.cache:
            return self.cache[query]
        
        try:
            # 1. Clasificar intención
            intent = self.classify_intent(query)
            
            # 2. Obtener embedding
            embedding = self.get_embedding(query)
            if not embedding:
                raise Exception("No se pudo obtener embedding")
            
            # 3. Buscar documentos
            contexts = self.search_documents(embedding)
            
            # 4. Generar respuesta
            response = self.generate_response(query, contexts)
            
            # 5. Calcular tiempo
            processing_time = time.time() - start_time
            
            result = {
                "response": response,
                "intent": intent,
                "sources_count": len(contexts),
                "processing_time": round(processing_time, 2),
                "success": True
            }
            
            # Guardar en cache
            if len(self.cache) < Config.CACHE_SIZE:
                self.cache[query] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return {
                "response": "Error procesando consulta",
                "intent": "error",
                "sources_count": 0,
                "processing_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }