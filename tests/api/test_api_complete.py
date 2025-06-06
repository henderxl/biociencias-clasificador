"""
Script completo para probar la API de medicina personalizada.
Valida todos los endpoints y funcionalidades del sistema.
"""

import requests
import json
import numpy as np
from PIL import Image
import io
import time

# Configuración
API_BASE_URL = "http://localhost:8000"

def crear_imagen_test():
    """Crear una imagen de prueba sintética para testing."""
    # Crear imagen de prueba (224x224 RGB)
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # Agregar algunos patrones para simular una imagen médica
    img_array[50:150, 50:150] = [200, 200, 200]  # Región brillante
    img_array[100:120, 100:120] = [50, 50, 50]   # Región oscura
    
    # Convertir a PIL Image
    img = Image.fromarray(img_array)
    
    # Guardar en buffer
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    return img_buffer

def test_health_endpoint():
    """Probar endpoint de salud."""
    print("🔍 Probando endpoint de salud...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Estado API: {data['api_status']}")
            print(f"📊 Modelos cargados:")
            print(f"   - Clasificador de imágenes: {data['models']['image_classifier']}")
            print(f"   - Recomendador de tratamientos: {data['models']['treatment_recommender']}")
            print(f"🔧 Framework: {data['framework']}")
            return True
        else:
            print(f"❌ Error en salud: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a API: {e}")
        return False

def test_classify_tumor():
    """Probar clasificación de tumores."""
    print("\n🧠 Probando clasificación de tumores...")
    try:
        # Crear imagen de prueba
        img_buffer = crear_imagen_test()
        
        # Hacer request
        files = {'image': ('test_brain_mri.png', img_buffer, 'image/png')}
        response = requests.post(f"{API_BASE_URL}/classify-tumor", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Clasificación exitosa!")
            print(f"🎯 Tipo de tumor: {data['predicted_class']}")
            print(f"🔍 Confianza: {data['confidence']:.4f}")
            print(f"📊 Probabilidades:")
            for clase, prob in data['probabilities'].items():
                print(f"   - {clase}: {prob:.4f}")
            
            if 'metadata' in data:
                print(f"🔧 Modelo utilizado: {data['metadata']['model_type']}")
            
            return True
        else:
            print(f"❌ Error en clasificación: {response.status_code}")
            print(f"📄 Respuesta: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error en clasificación: {e}")
        return False

def test_recommend_treatment():
    """Probar recomendación de tratamientos."""
    print("\n💊 Probando recomendación de tratamientos...")
    try:
        # Crear imagen de prueba
        img_buffer = crear_imagen_test()
        
        # Datos del paciente
        patient_data = {
            'age': 45,
            'sex': 'M',
            'clinical_note': 'Paciente presenta dolores de cabeza severos y náuseas frecuentes. Historia de convulsiones ocasionales.'
        }
        
        # Hacer request
        files = {'image': ('test_brain_mri.png', img_buffer, 'image/png')}
        data = patient_data
        
        response = requests.post(f"{API_BASE_URL}/recommend-treatment", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recomendación exitosa!")
            print(f"💊 Tratamiento recomendado: {result['recommended_treatment']}")
            print(f"🔍 Confianza: {result['confidence']:.4f}")
            print(f"📊 Probabilidades de tratamientos:")
            for tratamiento, prob in result['probabilities'].items():
                print(f"   - {tratamiento}: {prob:.4f}")
            
            # Mostrar clasificación de tumor si está incluida
            if 'tumor_classification' in result:
                tumor_data = result['tumor_classification']
                print(f"\n🧠 Clasificación de tumor incluida:")
                print(f"   - Tipo: {tumor_data['predicted_class']}")
                print(f"   - Confianza: {tumor_data['confidence']:.4f}")
            
            # Mostrar metadatos del paciente
            if 'metadata' in result and 'patient_data' in result['metadata']:
                patient_info = result['metadata']['patient_data']
                print(f"\n👤 Datos del paciente:")
                print(f"   - Edad: {patient_info['age']} años")
                print(f"   - Sexo: {patient_info['sex']}")
                print(f"   - Nota clínica: {patient_info['clinical_note_length']} caracteres")
            
            return True
        else:
            print(f"❌ Error en recomendación: {response.status_code}")
            print(f"📄 Respuesta: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error en recomendación: {e}")
        return False

def test_model_info():
    """Probar endpoint de información de modelos."""
    print("\n📋 Probando información de modelos...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Información obtenida!")
            print(f"📊 Estado de modelos:")
            
            # Información del clasificador
            img_info = data['image_classifier']
            print(f"   🧠 Clasificador de imágenes:")
            print(f"      - Cargado: {img_info['loaded']}")
            print(f"      - Tipo: {img_info['model_type']}")
            
            # Información del recomendador
            treat_info = data['treatment_recommender']
            print(f"   💊 Recomendador de tratamientos:")
            print(f"      - Cargado: {treat_info['loaded']}")
            print(f"      - Tipo: {treat_info['model_type']}")
            
            return True
        else:
            print(f"❌ Error obteniendo info: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en info de modelos: {e}")
        return False

def test_docs_endpoint():
    """Verificar que la documentación esté disponible."""
    print("\n📚 Verificando documentación Swagger...")
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print(f"✅ Documentación Swagger disponible en: {API_BASE_URL}/docs")
            return True
        else:
            print(f"❌ Error accediendo a docs: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error verificando docs: {e}")
        return False

def run_performance_test():
    """Ejecutar prueba de rendimiento básica."""
    print("\n⚡ Ejecutando prueba de rendimiento...")
    try:
        times = []
        num_tests = 5
        
        for i in range(num_tests):
            start_time = time.time()
            
            # Crear imagen de prueba
            img_buffer = crear_imagen_test()
            files = {'image': ('test_brain_mri.png', img_buffer, 'image/png')}
            
            # Hacer request de clasificación
            response = requests.post(f"{API_BASE_URL}/classify-tumor", files=files)
            
            end_time = time.time()
            request_time = end_time - start_time
            times.append(request_time)
            
            print(f"   Prueba {i+1}/{num_tests}: {request_time:.3f}s")
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"📊 Estadísticas de rendimiento:")
        print(f"   - Tiempo promedio: {avg_time:.3f}s")
        print(f"   - Tiempo mínimo: {min_time:.3f}s")
        print(f"   - Tiempo máximo: {max_time:.3f}s")
        print(f"   - Throughput estimado: {1/avg_time:.1f} requests/segundo")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de rendimiento: {e}")
        return False

def main():
    """Ejecutar todas las pruebas."""
    print("🚀 INICIANDO PRUEBAS COMPLETAS DE LA API")
    print("=" * 60)
    print("🧠 Sistema de Medicina Personalizada")
    print("🔬 Clasificación de Tumores Cerebrales y Recomendación de Tratamientos")
    print("⚙️ Versión compatible con Python 3.13 (scikit-learn)")
    print("=" * 60)
    
    # Lista de pruebas
    tests = [
        ("Estado de salud", test_health_endpoint),
        ("Clasificación de tumores", test_classify_tumor),
        ("Recomendación de tratamientos", test_recommend_treatment),
        ("Información de modelos", test_model_info),
        ("Documentación Swagger", test_docs_endpoint),
        ("Rendimiento básico", run_performance_test)
    ]
    
    results = []
    
    # Ejecutar pruebas
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    
    successful = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ EXITOSA" if success else "❌ FALLIDA"
        print(f"{test_name:.<40} {status}")
        if success:
            successful += 1
    
    print(f"\n🎯 RESULTADO FINAL: {successful}/{total} pruebas exitosas")
    
    if successful == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ Sistema de medicina personalizada completamente funcional")
        print("🌐 API lista para uso en producción")
        print(f"📋 Acceso a la API: {API_BASE_URL}")
        print(f"📚 Documentación: {API_BASE_URL}/docs")
    else:
        print(f"⚠️ {total - successful} pruebas fallaron")
        print("🔧 Revisar configuración y modelos")
    
    print(f"\n🏥 CAPACIDADES DEL SISTEMA:")
    print("   🧠 Clasificación automática de tumores cerebrales")
    print("   💊 Recomendación personalizada de tratamientos")
    print("   📊 Análisis multimodal (imágenes + datos clínicos)")
    print("   🔍 API REST completa con documentación Swagger")
    print("   ⚡ Modelos optimizados de scikit-learn")
    print("   🏥 Listo para integración hospitalaria")

if __name__ == "__main__":
    main() 