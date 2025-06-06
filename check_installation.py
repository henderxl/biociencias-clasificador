#!/usr/bin/env python3
"""
Script de Verificación de Instalación - Sistema de Medicina Personalizada
Verifica que todas las dependencias estén correctamente instaladas
"""

import sys
import importlib
import os

def print_header(title):
    print("\n" + "="*60)
    print(f"🔧 {title}")
    print("="*60)

def verificar_python():
    print("\n📋 VERIFICANDO PYTHON")
    print("-" * 30)
    
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✅ Versión de Python compatible")
        return True
    else:
        print("❌ Se requiere Python 3.9+ (recomendado 3.13+)")
        return False

def verificar_dependencias():
    print("\n📋 VERIFICANDO DEPENDENCIAS PRINCIPALES")
    print("-" * 40)
    
    dependencias_core = {
        'pandas': 'Análisis de datos',
        'numpy': 'Operaciones numéricas', 
        'sklearn': 'Machine Learning',
        'joblib': 'Serialización de modelos',
        'fastapi': 'API REST',
        'uvicorn': 'Servidor ASGI',
        'requests': 'Cliente HTTP'
    }
    
    exitos = 0
    total = len(dependencias_core)
    
    for paquete, descripcion in dependencias_core.items():
        try:
            module = importlib.import_module(paquete)
            version = getattr(module, '__version__', 'N/A')
            print(f"✅ {paquete} v{version} - {descripcion}")
            exitos += 1
        except ImportError:
            print(f"❌ {paquete} - {descripcion} (NO INSTALADO)")
    
    print(f"\n📊 Resultado: {exitos}/{total} dependencias instaladas")
    return exitos == total

def verificar_imagenes():
    print("\n📋 VERIFICANDO PROCESAMIENTO DE IMÁGENES")
    print("-" * 40)
    
    deps_img = {
        'cv2': ('opencv-python', 'Procesamiento de imágenes'),
        'PIL': ('Pillow', 'Manipulación de imágenes'),
        'matplotlib': ('matplotlib', 'Visualización')
    }
    
    exitos = 0
    for modulo, (paquete, desc) in deps_img.items():
        try:
            importlib.import_module(modulo)
            print(f"✅ {paquete} - {desc}")
            exitos += 1
        except ImportError:
            print(f"❌ {paquete} - {desc}")
    
    return exitos == len(deps_img)

def verificar_estructura():
    print("\n📋 VERIFICANDO ESTRUCTURA DEL PROYECTO")
    print("-" * 40)
    
    elementos = [
        ('data/', 'Directorio de datos'),
        ('models/', 'Modelos entrenados'),
        ('src/', 'Código fuente'),
        ('tests/', 'Tests automatizados'),
        ('requirements.txt', 'Dependencias'),
        ('api_server.py', 'Servidor API'),
        ('test_demo_simple.py', 'Demo funcional'),
        ('README.md', 'Documentación')
    ]
    
    exitos = 0
    for elemento, desc in elementos:
        if os.path.exists(elemento):
            print(f"✅ {elemento} - {desc}")
            exitos += 1
        else:
            print(f"❌ {elemento} - {desc}")
    
    return exitos == len(elementos)

def verificar_dataset():
    print("\n📋 VERIFICANDO DATASET")
    print("-" * 25)
    
    dataset_path = 'data/brain_conditions_detailed_dataset.csv'
    if os.path.exists(dataset_path):
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path, sep=';')
            print(f"✅ Dataset encontrado: {len(df)} registros")
            return True
        except Exception as e:
            print(f"❌ Error leyendo dataset: {e}")
            return False
    else:
        print(f"❌ Dataset no encontrado")
        return False

def verificar_tests():
    print("\n📋 VERIFICANDO TESTS")
    print("-" * 25)
    
    tests_disponibles = [
        ('tests/models/test_models_trained.py', 'Tests de modelos (139 líneas)'),
        ('tests/api/test_api_complete.py', 'Tests de API (292 líneas)')
    ]
    
    exitos = 0
    for test_path, desc in tests_disponibles:
        if os.path.exists(test_path):
            print(f"✅ {test_path} - {desc}")
            exitos += 1
        else:
            print(f"❌ {test_path} - {desc}")
    
    return exitos == len(tests_disponibles)

def main():
    print_header("VERIFICACIÓN DE INSTALACIÓN - MEDICINA PERSONALIZADA")
    print("🎯 Verificando que el sistema esté correctamente configurado...")
    
    # Ejecutar verificaciones
    resultados = {
        'Python': verificar_python(),
        'Dependencias Core': verificar_dependencias(),
        'Proc. Imágenes': verificar_imagenes(),
        'Estructura': verificar_estructura(),
        'Dataset': verificar_dataset(),
        'Tests': verificar_tests()
    }
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("="*60)
    
    total = len(resultados)
    exitosos = sum(resultados.values())
    
    for nombre, resultado in resultados.items():
        estado = "✅ OK" if resultado else "❌ FALTA"
        print(f"{nombre:<20}: {estado}")
    
    porcentaje = (exitosos / total) * 100
    print(f"\n📈 Completitud: {exitosos}/{total} ({porcentaje:.0f}%)")
    
    if porcentaje == 100:
        print("\n🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("🚀 Puedes ejecutar:")
        print("   • python test_demo_simple.py")
        print("   • python api_server.py")
        print("   • python tests/models/test_models_trained.py")
        print("   • python tests/api/test_api_complete.py")
    elif porcentaje >= 80:
        print("\n⚠️ Sistema mayormente funcional")
        print("💡 Corrige los elementos faltantes")
    else:
        print("\n❌ Sistema requiere configuración")
        print("🔧 Ejecutar: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 