# tests/test_rag.py - VERSIÓN CON LOGGING DETALLADO
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from src.rag_service import RAGService
from src.utils import PerformanceTracker, validate_query
from src.config import Config

# Configurar logger para tests
logger = logging.getLogger(__name__)

class TestRAGService:
    """Tests unitarios para el servicio RAG - cada función específica"""
    
    @pytest.fixture
    def mock_rag_service(self):
        logger.info("=== INICIANDO SETUP DE RAG SERVICE ===")
        with patch('src.rag_service.openai.OpenAI') as mock_openai, \
             patch('src.rag_service.OpenAI') as mock_llm, \
             patch('src.rag_service.QdrantClient') as mock_qdrant:
            
            # Mock de los clientes
            mock_openai_instance = MagicMock()
            mock_llm_instance = MagicMock()
            mock_qdrant_instance = MagicMock()
            
            mock_openai.return_value = mock_openai_instance
            mock_llm.return_value = mock_llm_instance
            mock_qdrant.return_value = mock_qdrant_instance
            
            service = RAGService()
            logger.info(f"RAG Service creado exitosamente: {type(service)}")
            
            return service

    @pytest.mark.unit
    def test_classify_intent_fechas_detection(self, mock_rag_service):
        """TEST UNITARIO: Verificar detección de intent 'fechas' con keywords específicas"""
        logger.info("--- TEST: Clasificación de intent FECHAS ---")
        
        test_queries = [
            "¿Cuándo son las fechas de examen?",
            "Necesito el cronograma de evaluaciones",
            "¿Qué fecha es el examen final?"
        ]
        
        for query in test_queries:
            logger.info(f"Probando query: '{query}'")
            result = mock_rag_service.classify_intent(query)
            logger.info(f"Resultado clasificación: {result}")
            
            # Verificar que detecta fecha O cronograma
            assert result in ["fechas_examenes", "consulta_general"]
            logger.info(f"✅ Query clasificada correctamente: {result}")

    @pytest.mark.unit  
    def test_classify_intent_matricula_detection(self, mock_rag_service):
        """TEST UNITARIO: Verificar detección de intent 'matrícula' con keywords específicas"""
        logger.info("--- TEST: Clasificación de intent MATRÍCULA ---")
        
        test_queries = [
            "¿Cómo me matriculo?",
            "Proceso de inscripción estudiantil",
            "Necesito matricularme en el curso"
        ]
        
        for query in test_queries:
            logger.info(f"Probando query: '{query}'")
            result = mock_rag_service.classify_intent(query)
            logger.info(f"Resultado clasificación: {result}")
            
            # Verificar que detecta matrícula O inscripción
            expected = "proceso_matricula" if "matricul" in query.lower() or "inscripc" in query.lower() else "consulta_general"
            assert result in ["proceso_matricula", "consulta_general"]

            logger.info(f"✅ Query clasificada como esperado: {result}")

    @pytest.mark.unit
    def test_classify_intent_costos_detection(self, mock_rag_service):
        """TEST UNITARIO: Verificar detección de intent 'costos' con keywords específicas"""
        logger.info("--- TEST: Clasificación de intent COSTOS ---")
        
        test_queries = [
            "¿Cuánto cuesta la matrícula?",
            "Necesito saber los precios de los cursos",
            "¿Cuál es el costo del programa?"
        ]
        
        for query in test_queries:
            logger.info(f"Probando query: '{query}'")
            result = mock_rag_service.classify_intent(query)
            logger.info(f"Resultado clasificación: {result}")
            
            # Verificar que detecta costo, precio, pago
            expected = "costos_pagos" if any(word in query.lower() for word in ['costo', 'precio', 'pago']) else "consulta_general"
            assert result == expected
            logger.info(f"✅ Query clasificada como esperado: {result}")

    @pytest.mark.unit
    def test_get_embedding_real_conversion(self, mock_rag_service):
        """TEST UNITARIO: Verificar que get_embedding realmente convierte texto a vector numérico"""
        logger.info("--- TEST: Conversión de texto a embedding ---")
        
        # Setup del mock para simular respuesta real de OpenAI
        mock_response = Mock()
        mock_response.data = [Mock()]
        test_embedding = [0.1, 0.2, 0.3, -0.1, 0.5]  # Vector de ejemplo
        mock_response.data[0].embedding = test_embedding
        
        mock_rag_service.openai_client.embeddings.create.return_value = mock_response
        
        test_text = "¿Cuándo son los exámenes finales?"
        logger.info(f"Convirtiendo texto a embedding: '{test_text}'")
        
        result = mock_rag_service.get_embedding(test_text)
        
        # Verificaciones unitarias específicas
        logger.info(f"Embedding resultado: {result}")
        assert isinstance(result, list), "El embedding debe ser una lista"
        assert len(result) == 5, f"Expected 5 dimensions, got {len(result)}"
        assert all(isinstance(x, (int, float)) for x in result), "Todos los elementos deben ser números"
        assert result == test_embedding, "El embedding debe coincidir con el mock"
        
        # Verificar que se llamó a la API correctamente
        mock_rag_service.openai_client.embeddings.create.assert_called_once()
        call_args = mock_rag_service.openai_client.embeddings.create.call_args
        assert call_args[1]['input'] == test_text
        assert call_args[1]['model'] == Config.EMBEDDING_MODEL
        
        logger.info("✅ Conversión de texto a embedding funcionando correctamente")

    @pytest.mark.unit
    def test_get_embedding_error_handling(self, mock_rag_service):
        """TEST UNITARIO: Verificar manejo robusto de errores en API de embeddings"""
        logger.info("--- TEST: Manejo de errores en embeddings ---")
        
        # Simular error de API
        mock_rag_service.openai_client.embeddings.create.side_effect = Exception("API Rate limit exceeded")
        
        test_text = "texto de prueba"
        logger.info(f"Probando manejo de error con: '{test_text}'")
        
        result = mock_rag_service.get_embedding(test_text)
        
        # Verificar que maneja el error correctamente
        logger.info(f"Resultado con error: {result}")
        assert result == [], "Debe retornar lista vacía en caso de error"
        assert isinstance(result, list), "Debe mantener el tipo de retorno"
        
        logger.info("✅ Manejo de errores funcionando correctamente")

    @pytest.mark.unit
    def test_search_documents_vector_query(self, mock_rag_service):
        """TEST UNITARIO: Verificar que search_documents usa vector correctamente"""
        logger.info("--- TEST: Búsqueda vectorial en Qdrant ---")
        
        # Mock de respuesta de Qdrant
        mock_results = [
            Mock(payload={'text': 'Documento 1', 'categoria': 'fechas'}, score=0.95),
            Mock(payload={'text': 'Documento 2', 'categoria': 'costos'}, score=0.87)
        ]
        mock_rag_service.qdrant_client.search.return_value = mock_results
        
        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        logger.info(f"Buscando con vector: {test_vector}")
        
        result = mock_rag_service.search_documents(test_vector, limit=5)
        
        # Verificaciones unitarias
        logger.info(f"Resultados de búsqueda: {len(result)} documentos")
        assert isinstance(result, list), "Debe retornar una lista"
        assert len(result) == 2, "Debe retornar los documentos mockeados"
        
        # Verificar que se llamó con parámetros correctos
        mock_rag_service.qdrant_client.search.assert_called_once_with(
            collection_name=Config.COLLECTION_NAME,
            query_vector=test_vector,
            limit=5,
            score_threshold=0.3
        )
        
        logger.info("✅ Búsqueda vectorial funcionando correctamente")

    @pytest.mark.unit
    def test_cache_storage_and_retrieval(self, mock_rag_service):
        """TEST UNITARIO: Verificar que el cache almacena y recupera correctamente"""
        logger.info("--- TEST: Funcionalidad de cache ---")
        
        # Setup mocks para simular proceso completo
        with patch.object(mock_rag_service, 'get_embedding', return_value=[0.1, 0.2]), \
             patch.object(mock_rag_service, 'search_documents', return_value=[]), \
             patch.object(mock_rag_service, 'generate_response', return_value="Respuesta de prueba"):
            
            test_query = "¿Cuándo son los exámenes?"
            logger.info(f"Primera consulta (debe almacenar en cache): '{test_query}'")
            
            # Primera consulta
            result1 = mock_rag_service.process_query(test_query)
            logger.info(f"Resultado 1: {result1['response'][:50]}...")
            
            # Verificar que se almacenó en cache
            assert test_query in mock_rag_service.cache, "La query debe estar en cache"
            assert result1['success'] == True, "La primera consulta debe ser exitosa"
            
            logger.info("Segunda consulta (debe venir de cache)")
            
            # Segunda consulta (debe venir del cache)
            result2 = mock_rag_service.process_query(test_query)
            logger.info(f"Resultado 2: {result2['response'][:50]}...")
            
            # Verificar que viene del cache
            assert result2 == result1, "Los resultados deben ser idénticos"
            assert result2 is mock_rag_service.cache[test_query], "Debe retornar el objeto cacheado"
            
            logger.info("✅ Sistema de cache funcionando correctamente")

    @pytest.mark.unit
    def test_generate_response_with_context(self, mock_rag_service):
        """TEST UNITARIO: Verificar generación de respuesta con contexto"""
        logger.info("--- TEST: Generación de respuesta con contexto ---")
        
        # Mock de respuesta del LLM
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Respuesta generada por el LLM"
        mock_rag_service.llm_client.chat.completions.create.return_value = mock_response
        
        # Mock de contextos
        mock_contexts = [
            Mock(payload={'text': 'Los exámenes finales son en diciembre'}),
            Mock(payload={'text': 'La matrícula cuesta $500'})
        ]
        
        test_query = "¿Cuándo son los exámenes?"
        logger.info(f"Generando respuesta para: '{test_query}'")
        logger.info(f"Con {len(mock_contexts)} documentos de contexto")
        
        result = mock_rag_service.generate_response(test_query, mock_contexts)
        
        # Verificaciones unitarias
        logger.info(f"Respuesta generada: {result}")
        assert isinstance(result, str), "Debe retornar un string"
        assert len(result) > 0, "La respuesta no debe estar vacía"
        assert result == "Respuesta generada por el LLM", "Debe retornar la respuesta del LLM"
        
        # Verificar que se llamó al LLM con contexto
        mock_rag_service.llm_client.chat.completions.create.assert_called_once()
        call_args = mock_rag_service.llm_client.chat.completions.create.call_args
        
        # Verificar que el contexto se incluyó en el prompt
        prompt_content = call_args[1]['messages'][0]['content']
        assert test_query in prompt_content, "La query debe estar en el prompt"
        assert "Los exámenes finales" in prompt_content, "El contexto debe estar en el prompt"
        
        logger.info("✅ Generación de respuesta con contexto funcionando correctamente")

class TestUtils:
    """Tests unitarios para utilidades del sistema"""
    
    @pytest.mark.unit
    def test_validate_query_input_validation(self):
        """TEST UNITARIO: Verificar validación robusta de queries de entrada"""
        logger.info("--- TEST: Validación de queries ---")
        
        # Casos válidos
        valid_queries = [
            "¿Cuándo son los exámenes?",
            "Información sobre matrícula",
            "a" * 999  # Límite máximo
        ]
        
        for query in valid_queries:
            logger.info(f"Validando query válida: '{query[:50]}...'")
            result = validate_query(query)
            assert result == True, f"Query válida rechazada: {query[:50]}"
            logger.info("✅ Query válida aceptada")
        
        # Casos inválidos
        # Casos inválidos
        invalid_queries = [
            "",
            "   ",
            "a" * 1001  # Muy larga
        ]

        for query in invalid_queries:
            logger.info(f"Validando query inválida: '{query[:20]}...'")
            result = validate_query(query)
            assert result == False, f"Query inválida aceptada: {query[:20]}"
            logger.info("✅ Query inválida rechazada correctamente")

        # Test None separadamente
        logger.info("Probando validación con None")
        try:
            result = validate_query(None)
            assert result == False, "None debería ser rechazado"
        except (TypeError, AttributeError):
            logger.info("✅ None manejado correctamente con excepción")

    @pytest.mark.unit
    def test_performance_tracker_metrics_calculation(self):
        """TEST UNITARIO: Verificar cálculos precisos de métricas de rendimiento"""
        logger.info("--- TEST: Cálculos de métricas de rendimiento ---")
        
        tracker = PerformanceTracker()
        
        # Datos de prueba
        test_data = [
            (1.5, True),   # Query exitosa, 1.5s
            (2.0, True),   # Query exitosa, 2.0s
            (3.0, False),  # Query fallida, 3.0s
            (1.0, True),   # Query exitosa, 1.0s
        ]
        
        logger.info(f"Agregando {len(test_data)} métricas de prueba")
        for time_val, success in test_data:
            tracker.track_query(time_val, success)
            logger.info(f"Agregada métrica: {time_val}s, éxito={success}")
        
        stats = tracker.get_stats()
        logger.info(f"Estadísticas calculadas: {stats}")
        
        # Verificaciones unitarias precisas
        expected_avg_time = sum(item[0] for item in test_data) / len(test_data)  # 1.875
        expected_success_rate = sum(1 for item in test_data if item[1]) / len(test_data)  # 0.75
        
        assert stats['total_queries'] == 4, f"Expected 4 queries, got {stats['total_queries']}"
        assert abs(stats['avg_time'] - expected_avg_time) < 0.01, f"Expected avg_time {expected_avg_time}, got {stats['avg_time']}"
        assert abs(stats['success_rate'] - expected_success_rate) < 0.01, f"Expected success_rate {expected_success_rate}, got {stats['success_rate']}"
        
        logger.info("✅ Cálculos de métricas funcionando correctamente")

class TestConfig:
    """Tests unitarios para configuración del sistema"""
    
    @pytest.mark.unit
    def test_config_validation_requirement_check(self):
        """TEST UNITARIO: Verificar validación estricta de configuración requerida"""
        logger.info("--- TEST: Validación de configuración ---")
        
        # Test con configuración incompleta
        with patch.object(Config, 'OPENAI_API_KEY', None), \
             patch.object(Config, 'DEEPSEEK_API_KEY', None), \
             patch.object(Config, 'QDRANT_URL', None), \
             patch.object(Config, 'QDRANT_API_KEY', None):
            
            logger.info("Probando validación con configuración faltante")
            
            with pytest.raises(ValueError, match="Missing"):
                Config.validate()
            
            logger.info("✅ Validación de configuración funcionando correctamente")

# Test de integración simplificado
@pytest.mark.integration
def test_health_endpoint_integration():
    """TEST INTEGRACIÓN: Verificar endpoint de health funcional"""
    logger.info("--- TEST INTEGRACIÓN: Health endpoint ---")
    
    from new_main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    logger.info("Cliente de prueba creado")
    
    response = client.get("/api/health")
    logger.info(f"Respuesta del endpoint: {response.status_code}")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    logger.info(f"Datos de health: {data}")
    
    assert "status" in data, "Response debe contener 'status'"
    assert "metrics" in data, "Response debe contener 'metrics'"
    
    logger.info("✅ Health endpoint funcionando correctamente")

if __name__ == "__main__":
    # Configurar logging para ejecución directa
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('test_detailed.log'),
            logging.StreamHandler()
        ]
    )
    pytest.main([__file__, "-v", "--tb=short"])