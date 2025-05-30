from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import json
import openai
from openai import OpenAI
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional, Generator
import os
from dotenv import load_dotenv
import uvicorn
from datetime import datetime
import numpy as np
from collections import defaultdict
import hashlib
import time
import asyncio

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="RAG UNI FIEE - Streaming API",
    description="API RAG con streaming responses y clasificación puramente semántica",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    stream: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    message_type: str
    sources: List[Dict[str, Any]] = []
    session_id: str
    quality_score: float = 0.0
    search_strategy: str = "semantic"

class QueryCache:
    """Cache inteligente para consultas similares"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        
    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
        
    def get(self, query: str) -> Optional[Dict]:
        key = self._get_cache_key(query)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def set(self, query: str, result: Dict):
        key = self._get_cache_key(query)
        
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = result
        self.access_times[key] = time.time()

class ChatSession:
    def __init__(self):
        self.sessions = {}
        
    def get_history(self, session_id: str) -> List[str]:
        return self.sessions.get(session_id, [])
    
    def add_to_history(self, session_id: str, user_message: str, assistant_response: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].extend([user_message, assistant_response])
        
        if len(self.sessions[session_id]) > 12:
            self.sessions[session_id] = self.sessions[session_id][-12:]
    
    def format_history(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return "Primera conversación"
        
        formatted = []
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                formatted.append(f"Usuario: {history[i]}")
                formatted.append(f"Asistente: {history[i + 1]}")
        
        return "\n".join(formatted[-6:])

class SemanticRAGChatbot:
    """Sistema RAG puramente semántico con streaming responses"""
    
    def __init__(self):
        # Configuración
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not all([self.openai_api_key, self.deepseek_api_key, self.qdrant_url, self.qdrant_api_key]):
            raise ValueError("Faltan variables de entorno necesarias")
        
        # Clientes
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.llm_client = OpenAI(
            api_key=self.deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Configuración
        self.collection_name = "uni_academic_system"
        self.embedding_model = "text-embedding-3-small"
        
        # Componentes
        self.session_manager = ChatSession()
        self.query_cache = QueryCache()
        
        print("✅ Sistema RAG semántico con streaming inicializado")

    def classify_intent_semantic(self, query: str, context: str = "") -> str:
        """Clasificación puramente semántica usando embeddings"""
        try:
            # Crear embeddings para categorías de referencia
            category_examples = {
                'fechas_examenes': "fechas exámenes finales cronograma calendario académico evaluaciones cuando cuándo",
                'proceso_matricula': "matrícula inscripción registro proceso pasos procedimiento cómo matricularse",
                'costos_pagos': "costos precios aranceles pagos tarifas cuánto cuesta dinero",
                'ubicacion_tramites': "ubicación oficina dirección dónde lugar sede",
                'requisitos_documentos': "requisitos documentos certificados constancias papeles qué necesito",
                'conversacion_casual': "hola gracias chau buenos días cómo estás saludo despedida",
                'consulta_general': "información general universidad FIEE académico"
            }
            
            # Generar embedding de la query
            query_embedding = self.openai_client.embeddings.create(
                input=query,
                model=self.embedding_model
            ).data[0].embedding
            
            # Calcular similitud con cada categoría
            max_similarity = -1
            best_intent = 'consulta_general'
            
            for intent, example_text in category_examples.items():
                category_embedding = self.openai_client.embeddings.create(
                    input=example_text,
                    model=self.embedding_model
                ).data[0].embedding
                
                # Calcular similitud coseno
                similarity = np.dot(query_embedding, category_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(category_embedding)
                )
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_intent = intent
            
            print(f"🧠 Clasificación semántica: {best_intent} (similitud: {max_similarity:.3f})")
            return best_intent
            
        except Exception as e:
            print(f"Error en clasificación semántica: {e}")
            return 'consulta_general'

    def semantic_search_pure(self, query: str, intent: str, limit: int = 8) -> List[Dict]:
        """Búsqueda semántica pura sin sesgos"""
        try:
            # Generar embedding directamente de la query original
            response = self.openai_client.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            query_embedding = response.data[0].embedding
            
            # Umbrales adaptativos pero conservadores
            intent_thresholds = {
                'fechas_examenes': 0.35,
                'proceso_matricula': 0.4,
                'costos_pagos': 0.45,
                'ubicacion_tramites': 0.4,
                'requisitos_documentos': 0.4,
                'conversacion_casual': 0.5,
                'consulta_general': 0.35
            }
            
            threshold = intent_thresholds.get(intent, 0.4)
            
            print(f"🔍 Búsqueda semántica pura - Umbral: {threshold:.2f}")
            
            # Búsqueda en Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=threshold
            )
            
            # Si no hay resultados, intentar con umbral más bajo
            if not results:
                print("🔍 Reintentando con umbral más bajo...")
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=max(0.2, threshold - 0.15)
                )
            
            print(f"✅ Encontrados {len(results)} documentos relevantes")
            
            # Formatear resultados
            contexts = []
            for point in results:
                contexts.append({
                    'score': point.score,
                    'text': point.payload['text'],
                    'metadata': point.payload['metadata'],
                    'type': point.payload['type']
                })
                
            return contexts
            
        except Exception as e:
            print(f"Error en búsqueda semántica: {e}")
            return []

    def stream_response_generator(self, query: str, contexts: List[Dict], intent: str, session_id: str) -> Generator[str, None, None]:
        """Generador de respuesta streaming"""
        try:
            if intent == 'conversacion_casual':
                yield from self.stream_casual_response(query)
                return
            
            if not contexts:
                yield from self.stream_no_context_response(query, intent)
                return
            
            # Preparar contextos para el prompt
            top_contexts = contexts[:3]
            formatted_contexts = []
            
            for i, ctx in enumerate(top_contexts):
                source = ctx['metadata'].get('categoria', 'Documento')
                text = ctx['text'][:500]
                formatted_contexts.append(f"Fuente {i+1} - {source}:\n{text}")
            
            contexts_text = "\n\n".join(formatted_contexts)
            
            # Prompt para streaming
            system_prompt = """Eres un asistente especializado de la FIEE-UNI. Responde usando únicamente la información proporcionada.

INSTRUCCIONES:
- Responde de manera profesional pero amigable
- Usa SOLO información del contexto proporcionado
- Para fechas: menciona fechas específicas y plazos
- Para procesos: da pasos claros y ordenados  
- Para costos: especifica montos exactos
- Usa emojis para organizar (📅 fechas, 💰 costos, 📋 pasos, 📍 ubicación)
- Si la información no está completa, dilo claramente"""

            user_prompt = f"""INFORMACIÓN DISPONIBLE:
{contexts_text}

CONSULTA DEL ESTUDIANTE: "{query}"

Responde de manera útil y específica:"""

            # Stream de respuesta
            stream = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=600,
                stream=True  # ¡STREAMING HABILITADO!
            )
            
            # Enviar chunks en tiempo real
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content, 'type': 'content'})}\n\n"
            
            # Enviar sources al final
            sources_data = [
                {
                    "categoria": ctx['metadata'].get('categoria', 'N/A'),
                    "tipo": ctx['type'],
                    "score": round(ctx['score'], 3),
                    "extracto": ctx['text'][:100] + "..." if len(ctx['text']) > 100 else ctx['text']
                }
                for ctx in contexts[:3]
            ]
            
            yield f"data: {json.dumps({'sources': sources_data, 'type': 'sources'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            error_msg = f"Disculpa, tuve un problema procesando tu consulta: {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def stream_casual_response(self, query: str) -> Generator[str, None, None]:
        """Stream para respuestas casuales"""
        try:
            casual_prompt = f"""Responde como un asistente amigable de la FIEE-UNI.

MENSAJE: "{query}"

Responde de manera natural, amigable y conversacional. Usa emojis ocasionalmente.
No uses listas ni bullet points. Mantén un tono cercano y empático."""

            stream = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": casual_prompt}],
                temperature=0.8,
                max_tokens=200,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content, 'type': 'content'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            fallback_msg = "¡Hola! 😊 ¿En qué puedo ayudarte hoy?"
            yield f"data: {json.dumps({'content': fallback_msg, 'type': 'content'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def stream_no_context_response(self, query: str, intent: str) -> Generator[str, None, None]:
        """Stream cuando no hay contexto relevante"""
        response_parts = [
            f'No encontré información específica sobre "{query}" en mi base de conocimientos. 📚\n\n',
            "Para ayudarte mejor, podrías intentar:\n",
            "• Ser más específico en tu consulta\n",
            "• Usar términos académicos más precisos\n",
            "• Reformular tu pregunta de otra manera\n\n",
            "También puedes contactar directamente:\n",
            "📞 **DIRCE** - Para trámites académicos\n",
            "📞 **Secretaría FIEE** - Para consultas de la facultad\n",
            "📞 **Mesa de Partes** - Para información general\n\n",
            "¿Hay algo más específico en lo que pueda ayudarte? 😊"
        ]
        
        for part in response_parts:
            yield f"data: {json.dumps({'content': part, 'type': 'content'})}\n\n"
            # Pequeña pausa para efecto de escritura
            time.sleep(0.1)
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def consultar_streaming(self, mensaje: str, session_id: str) -> Generator[str, None, None]:
        """Método principal con streaming response"""
        start_time = time.time()
        
        # Cache check
        cached_result = self.query_cache.get(mensaje)
        if cached_result:
            print("⚡ Respuesta desde cache")
            # Simular streaming del cache
            cached_response = cached_result["response"]
            words = cached_response.split()
            for i in range(0, len(words), 3):  # 3 palabras por chunk
                chunk = " ".join(words[i:i+3]) + " "
                yield f"data: {json.dumps({'content': chunk, 'type': 'content'})}\n\n"
                time.sleep(0.05)  # Simular velocidad de escritura
            
            if cached_result["sources"]:
                yield f"data: {json.dumps({'sources': cached_result['sources'], 'type': 'sources'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        
        # Obtener contexto de sesión
        context = self.session_manager.format_history(session_id)
        
        # 1. Clasificación semántica pura
        intent = self.classify_intent_semantic(mensaje, context)
        yield f"data: {json.dumps({'status': f'Analizando consulta... ({intent})', 'type': 'status'})}\n\n"
        
        # 2. Búsqueda semántica
        yield f"data: {json.dumps({'status': 'Buscando información relevante...', 'type': 'status'})}\n\n"
        contexts = self.semantic_search_pure(mensaje, intent, limit=8)
        
        # 3. Generar respuesta streaming
        yield f"data: {json.dumps({'status': 'Generando respuesta...', 'type': 'status'})}\n\n"
        
        # Recopilar respuesta completa para cache
        full_response = ""
        sources_data = []
        
        # Stream la respuesta y recopilar para cache
        for chunk_data in self.stream_response_generator(mensaje, contexts, intent, session_id):
            yield chunk_data
            
            # Extraer contenido para cache
            if 'data: ' in chunk_data:
                try:
                    chunk_json = json.loads(chunk_data.replace('data: ', '').strip())
                    if chunk_json.get('type') == 'content':
                        full_response += chunk_json.get('content', '')
                    elif chunk_json.get('type') == 'sources':
                        sources_data = chunk_json.get('sources', [])
                except:
                    pass
        
        # Guardar en cache para futuras consultas
        if full_response:
            cache_result = {
                "response": full_response.strip(),
                "message_type": "casual" if intent == 'conversacion_casual' else "tecnica",
                "sources": sources_data,
                "session_id": session_id,
                "quality_score": round(np.mean([c['score'] for c in contexts]) if contexts else 0.0, 3),
                "search_strategy": "semantic_streaming"
            }
            self.query_cache.set(mensaje, cache_result)
            
            # Actualizar historial
            self.session_manager.add_to_history(session_id, mensaje, full_response.strip())
        
        response_time = time.time() - start_time
        print(f"⚡ Stream completado en {response_time:.2f}s")

    def consultar_non_streaming(self, mensaje: str, session_id: str) -> Dict[str, Any]:
        """Método tradicional para compatibilidad"""
        # Cache check
        cached_result = self.query_cache.get(mensaje)
        if cached_result:
            return cached_result
        
        context = self.session_manager.format_history(session_id)
        intent = self.classify_intent_semantic(mensaje, context)
        contexts = self.semantic_search_pure(mensaje, intent, limit=8)
        
        # Generar respuesta completa
        if intent == 'conversacion_casual':
            try:
                response = self.llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": f'Responde amigablemente como asistente de FIEE-UNI: "{mensaje}". Máximo 200 palabras.'}],
                    temperature=0.8,
                    max_tokens=300
                )
                full_response = response.choices[0].message.content
            except:
                full_response = "¡Hola! 😊 Soy tu asistente de FIEE-UNI. ¿En qué puedo ayudarte?"
        elif not contexts:
            full_response = f'No encontré información específica sobre "{mensaje}". ¿Podrías ser más específico?'
        else:
            # Generar respuesta con contextos
            top_contexts = contexts[:3]
            formatted_contexts = []
            for i, ctx in enumerate(top_contexts):
                source = ctx['metadata'].get('categoria', 'Documento')
                text = ctx['text'][:500]
                formatted_contexts.append(f"Fuente {i+1} - {source}:\n{text}")
            
            contexts_text = "\n\n".join(formatted_contexts)
            
            try:
                response = self.llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{
                        "role": "user", 
                        "content": f"Usando esta información:\n{contexts_text}\n\nResponde a: '{mensaje}'"
                    }],
                    temperature=0.3,
                    max_tokens=600
                )
                full_response = response.choices[0].message.content
            except Exception as e:
                full_response = "Disculpa, tuve un problema procesando tu consulta."
        
        # Formatear resultado
        result = {
            "response": full_response,
            "message_type": "casual" if intent == 'conversacion_casual' else "tecnica",
            "sources": [
                {
                    "categoria": ctx['metadata'].get('categoria', 'N/A'),
                    "tipo": ctx['type'],
                    "score": round(ctx['score'], 3),
                    "extracto": ctx['text'][:100] + "..." if len(ctx['text']) > 100 else ctx['text']
                }
                for ctx in contexts[:3]
            ],
            "session_id": session_id,
            "quality_score": round(np.mean([c['score'] for c in contexts]) if contexts else 0.0, 3),
            "search_strategy": "semantic_pure"
        }
        
        # Cache y historial
        self.query_cache.set(mensaje, result)
        self.session_manager.add_to_history(session_id, mensaje, full_response)
        
        return result

# Inicializar sistema
try:
    chatbot = SemanticRAGChatbot()
    print("✅ Sistema RAG semántico con streaming inicializado correctamente")
except Exception as e:
    print(f"❌ Error inicializando chatbot: {e}")
    chatbot = None

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Servir tu frontend HTML personalizado"""
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        # Frontend de respaldo si no encuentra index.html
        html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG UNI - Asistente Académico</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #8B0000;      /* Guinda UNI */
            --primary-dark: #5F0000;       /* Guinda oscuro */
            --secondary-color: #f8fafc;
            --accent-color: #CD853F;       /* Dorado/beige */
            --text-color: #1f2937;
            --text-light: #6b7280;
            --border-color: #e5e7eb;
            --error-color: #ef4444;
            --warning-color: #f59e0b;
            --success-color: #10b981;
            --white: #ffffff;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #8B0000 0%, #5F0000 50%, #2D0000 100%);
            min-height: 100vh;
            color: var(--text-color);
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 30px 20px;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            border: 2px solid var(--primary-color);
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        .header p {
            font-size: 1rem;
            color: var(--text-light);
            max-width: 600px;
            margin: 0 auto;
        }

        .logo {
            width: 55px;
            height: 55px;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 26px;
            box-shadow: 0 8px 20px rgba(139, 0, 0, 0.3);
        }

        .main-content {
            display: flex;
            justify-content: center;
            flex: 1;
        }

        .chat-container {
            background: rgba(255, 255, 255, 0.98);
            border-radius: 25px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            height: 650px;
            width: 100%;
            max-width: 800px;
            overflow: hidden;
            border: 3px solid var(--primary-color);
        }

        .chat-header {
            padding: 25px 30px;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            border-radius: 22px 22px 0 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .chat-header h3 {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.3rem;
            font-weight: 600;
        }

        .chat-messages {
            flex: 1;
            padding: 25px;
            overflow-y: auto;
            scroll-behavior: smooth;
            background: linear-gradient(180deg, #fefefe 0%, #f9f9f9 100%);
        }

        .message {
            display: flex;
            margin-bottom: 25px;
            animation: slideIn 0.4s ease-out;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 75%;
            padding: 18px 24px;
            border-radius: 20px;
            position: relative;
            line-height: 1.5;
            font-size: 15px;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            border-bottom-right-radius: 8px;
            box-shadow: 0 4px 15px rgba(139, 0, 0, 0.3);
        }

        .message.assistant .message-content {
            background: var(--white);
            color: var(--text-color);
            border-bottom-left-radius: 8px;
            border: 2px solid #f0f0f0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .message-avatar {
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 12px;
            font-size: 18px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .message.user .message-avatar {
            background: linear-gradient(135deg, var(--accent-color), #B8860B);
            color: white;
            order: 2;
        }

        .message.assistant .message-avatar {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
        }

        .chat-input-container {
            padding: 25px;
            background: white;
            border-top: 2px solid var(--primary-color);
            border-radius: 0 0 22px 22px;
        }

        .chat-input-form {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .chat-input {
            flex: 1;
            padding: 18px 25px;
            border: 2px solid var(--primary-color);
            border-radius: 50px;
            font-size: 16px;
            outline: none;
            transition: all 0.3s ease;
            background: white;
        }

        .chat-input:focus {
            border-color: var(--primary-dark);
            box-shadow: 0 0 0 4px rgba(139, 0, 0, 0.1);
            transform: translateY(-1px);
        }

        .send-button {
            width: 55px;
            height: 55px;
            border: none;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(139, 0, 0, 0.3);
        }

        .send-button:hover {
            transform: scale(1.05) translateY(-2px);
            box-shadow: 0 8px 25px rgba(139, 0, 0, 0.4);
        }

        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--primary-color);
            font-style: italic;
            font-weight: 500;
        }

        .loading-dots {
            display: flex;
            gap: 4px;
        }

        .loading-dots span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--primary-color);
            animation: bounce 1.4s ease-in-out infinite both;
        }

        .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
        .loading-dots span:nth-child(2) { animation-delay: -0.16s; }

        .error-message {
            background: #fef2f2;
            color: var(--error-color);
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #fecaca;
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .streaming-indicator {
            display: inline-block;
            width: 8px;
            height: 12px;
            background: var(--primary-color);
            animation: pulse 1s infinite;
            margin-left: 4px;
        }

        .sources-info {
            margin-top: 15px;
            padding: 10px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
            font-size: 13px;
            color: var(--text-light);
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        @keyframes bounce {
            0%, 80%, 100% {
                transform: scale(0);
            }
            40% {
                transform: scale(1);
            }
        }

        @media (max-width: 768px) {
            .main-content {
                justify-content: center;
            }

            .header h1 {
                font-size: 1.8rem;
            }

            .chat-container {
                height: 550px;
                margin: 0 10px;
            }

            .container {
                padding: 15px;
            }

            .chat-input {
                font-size: 16px;
            }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header fade-in">
            <h1>
                <div class="logo">
                    <i class="fas fa-graduation-cap"></i>
                </div>
                RAG UNI - Asistente Académico (Streaming)
            </h1>
            <p>Tu asistente inteligente con respuestas en tiempo real para consultas académicas de la Universidad Nacional de Ingeniería</p>
        </div>

        <div class="main-content">
            <div class="chat-container fade-in">
                <div class="chat-header">
                    <h3>
                        <i class="fas fa-graduation-cap"></i>
                        Asistente Académico FIEE-UNI ⚡
                    </h3>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="message-content">
                            ¡Hola! 👋 Soy tu asistente académico de la FIEE-UNI con <strong>respuestas en tiempo real</strong>. Estoy aquí para ayudarte con:
                            <br><br>
                            📅 <strong>Cronogramas y fechas importantes</strong><br>
                            📋 <strong>Trámites y procedimientos académicos</strong><br>
                            💰 <strong>Costos y formas de pago</strong><br>
                            📚 <strong>Requisitos y documentación</strong><br>
                            🎓 <strong>Procesos de matrícula y traslados</strong><br>
                            <br>
                            ⚡ <strong>¡Verás mi respuesta aparecer en tiempo real!</strong><br>
                            ¿En qué puedo ayudarte hoy?
                        </div>
                    </div>
                </div>

                <div class="chat-input-container">
                    <form class="chat-input-form" id="chatForm">
                        <input 
                            type="text" 
                            class="chat-input" 
                            id="chatInput" 
                            placeholder="Pregunta sobre fechas, trámites, costos, etc..."
                            autocomplete="off"
                        >
                        <button type="submit" class="send-button" id="sendButton">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script>
        class StreamingRAGChatbot {
            constructor() {
                this.chatForm = document.getElementById('chatForm');
                this.chatInput = document.getElementById('chatInput');
                this.chatMessages = document.getElementById('chatMessages');
                this.sendButton = document.getElementById('sendButton');
                
                this.initializeEventListeners();
                this.isLoading = false;
                this.currentStreamingMessage = null;
            }

            initializeEventListeners() {
                this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));

                this.chatInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.handleSubmit(e);
                    }
                });

                this.chatInput.focus();
            }

            async handleSubmit(e) {
                e.preventDefault();
                
                const message = this.chatInput.value.trim();
                if (!message || this.isLoading) return;

                // Mostrar mensaje del usuario
                this.addMessage(message, 'user');
                this.chatInput.value = '';
                this.setLoading(true);

                try {
                    await this.sendStreamingMessage(message);
                } catch (error) {
                    this.addMessage('Lo siento, hubo un error al procesar tu consulta. Por favor, intenta nuevamente. 😔', 'assistant', true);
                    console.error('Error:', error);
                } finally {
                    this.setLoading(false);
                    this.chatInput.focus();
                }
            }

            async sendStreamingMessage(message) {
                try {
                    const response = await fetch('/api/chat/stream', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: 'web_session'
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`Error ${response.status}: ${response.statusText}`);
                    }

                    // Crear mensaje vacío para streaming
                    this.currentStreamingMessage = this.createStreamingMessage();
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    this.handleStreamingData(data);
                                } catch (e) {
                                    console.error('Error parsing streaming data:', e);
                                }
                            }
                        }
                    }

                } catch (error) {
                    console.error('Streaming error:', error);
                    throw error;
                }
            }

            createStreamingMessage() {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';
                
                const avatar = document.createElement('div');
                avatar.className = 'message-avatar';
                avatar.innerHTML = '<i class="fas fa-robot"></i>';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContent.innerHTML = '<span class="streaming-indicator"></span>';
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(messageContent);
                
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
                
                return messageContent;
            }

            handleStreamingData(data) {
                if (!this.currentStreamingMessage) return;

                switch (data.type) {
                    case 'status':
                        // Mostrar status de procesamiento
                        this.currentStreamingMessage.innerHTML = `<div class="loading"><i class="fas fa-brain"></i> ${data.status}</div>`;
                        break;
                        
                    case 'content':
                        // Agregar contenido en tiempo real
                        if (this.currentStreamingMessage.innerHTML.includes('streaming-indicator') || 
                            this.currentStreamingMessage.innerHTML.includes('loading')) {
                            this.currentStreamingMessage.innerHTML = '';
                        }
                        this.currentStreamingMessage.innerHTML += data.content;
                        this.scrollToBottom();
                        break;
                        
                    case 'sources':
                        // Agregar información de fuentes
                        if (data.sources && data.sources.length > 0) {
                            const sourcesHtml = `
                                <div class="sources-info">
                                    <i class="fas fa-book"></i> <strong>Fuentes consultadas:</strong> ${data.sources.length} documentos relevantes
                                </div>
                            `;
                            this.currentStreamingMessage.innerHTML += sourcesHtml;
                        }
                        break;
                        
                    case 'done':
                        // Streaming completado
                        this.currentStreamingMessage = null;
                        this.scrollToBottom();
                        break;
                        
                    case 'error':
                        // Error en streaming
                        this.currentStreamingMessage.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-triangle"></i>${data.content}</div>`;
                        this.currentStreamingMessage = null;
                        break;
                }
            }

            addMessage(content, sender, isError = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const avatar = document.createElement('div');
                avatar.className = 'message-avatar';
                avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                
                if (isError) {
                    messageContent.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-triangle"></i>${content}</div>`;
                } else {
                    messageContent.innerHTML = content;
                }
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(messageContent);
                
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }

            setLoading(loading) {
                this.isLoading = loading;
                this.sendButton.disabled = loading;
                
                if (loading) {
                    this.sendButton.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
                } else {
                    this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
                }
            }

            scrollToBottom() {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }
        }

        // Inicializar el chatbot streaming cuando el DOM esté listo
        document.addEventListener('DOMContentLoaded', () => {
            new StreamingRAGChatbot();
        });
    </script>
</body>
</html>
        """
        return HTMLResponse(content=html_content, status_code=200)

@app.post("/api/chat/stream")
async def chat_stream_endpoint(message: ChatMessage):
    """Endpoint principal con streaming responses"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot no inicializado")
    
    def generate():
        try:
            for chunk in chatbot.consultar_streaming(message.message, message.session_id):
                yield chunk
        except Exception as e:
            error_chunk = f"data: {json.dumps({'content': f'Error: {str(e)}', 'type': 'error'})}\n\n"
            yield error_chunk
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Endpoint tradicional para compatibilidad"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot no inicializado")
    
    try:
        if message.stream:
            # Redirigir a streaming
            raise HTTPException(status_code=307, detail="Use /api/chat/stream for streaming responses")
        else:
            resultado = chatbot.consultar_non_streaming(message.message, message.session_id)
            return ChatResponse(**resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/health")
async def health_check():
    try:
        if not chatbot:
            return {"status": "error", "error": "Chatbot no inicializado"}
            
        collections = chatbot.qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if chatbot.collection_name in collection_names:
            collection_info = chatbot.qdrant_client.get_collection(chatbot.collection_name)
            status = "healthy"
            points_count = collection_info.points_count
        else:
            status = "no_collection"
            points_count = 0
            
        return {
            "status": status,
            "collection": chatbot.collection_name,
            "documents": points_count,
            "active_sessions": len(chatbot.session_manager.sessions),
            "cache_size": len(chatbot.query_cache.cache),
            "features": {
                "semantic_classification": True,
                "streaming_responses": True,
                "pure_semantic_search": True,
                "smart_caching": True,
                "no_keyword_bias": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/clear_session")
async def clear_session(session_data: dict):
    session_id = session_data.get("session_id", "default")
    
    if chatbot and session_id in chatbot.session_manager.sessions:
        del chatbot.session_manager.sessions[session_id]
        return {"message": f"Sesión {session_id} limpiada"}
    return {"message": "Sesión no encontrada"}

@app.get("/api/stats")
async def get_stats():
    """Estadísticas del sistema"""
    if not chatbot:
        return {"error": "Chatbot no disponible"}
        
    try:
        collection_info = chatbot.qdrant_client.get_collection(chatbot.collection_name)
        
        return {
            "total_documents": collection_info.points_count,
            "collection_status": collection_info.status,
            "embedding_model": chatbot.embedding_model,
            "active_sessions": len(chatbot.session_manager.sessions),
            "cache_entries": len(chatbot.query_cache.cache),
            "system_type": "semantic_streaming_rag",
            "features": {
                "streaming": True,
                "semantic_only": True,
                "no_keywords": True,
                "real_time_response": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Iniciando Sistema RAG Semántico con Streaming...")
    print("🎓 FIEE-UNI - Sin Keywords, Solo Semántica")
    print("⚡ Con Streaming Responses en Tiempo Real")
    print("🌐 Frontend disponible en http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )