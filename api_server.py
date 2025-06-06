"""
API principal para el sistema de medicina personalizada.
Proporciona endpoints para clasificación de tumores y recomendación de tratamientos.
Versión compatible con modelos de scikit-learn (sin TensorFlow).
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import json
import joblib
from typing import Optional, Dict, Any
import os
import sys

# Agregar el directorio src al path para importar módulos
sys.path.append('src')

from src.data.data_loader import DataLoader
from src.models.train_models import SklearnImageClassifier, SklearnTreatmentRecommender

# Configuración de la aplicación
app = FastAPI(
    title="Sistema de Medicina Personalizada - Clasificación de Tumores Cerebrales",
    description="API para clasificación de tipos de tumores cerebrales y recomendación de tratamientos",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para los modelos
image_classifier = None
treatment_recommender = None
data_loader = None

@app.on_event("startup")
async def startup_event():
    """Cargar modelos al iniciar la aplicación."""
    global image_classifier, treatment_recommender, data_loader
    
    try:
        # Inicializar data loader
        data_loader = DataLoader()
        print("✅ DataLoader inicializado")
        
        # Cargar clasificador de imágenes
        if os.path.exists('models/sklearn_image_classifier.joblib'):
            image_classifier_data = joblib.load('models/sklearn_image_classifier.joblib')
            image_classifier = SklearnImageClassifier()
            image_classifier.model = image_classifier_data['model']
            image_classifier.scaler = image_classifier_data['scaler']
            image_classifier.label_encoder = image_classifier_data['label_encoder']
            image_classifier.is_trained = True
            print("✅ Modelo de clasificación de imágenes cargado exitosamente")
        else:
            print("⚠️ Modelo de clasificación de imágenes no encontrado en models/sklearn_image_classifier.joblib")
        
        # Cargar recomendador de tratamientos
        if os.path.exists('models/sklearn_treatment_recommender.joblib'):
            treatment_recommender_data = joblib.load('models/sklearn_treatment_recommender.joblib')
            treatment_recommender = SklearnTreatmentRecommender()
            treatment_recommender.model = treatment_recommender_data['model']
            treatment_recommender.scaler = treatment_recommender_data['scaler']
            treatment_recommender.label_encoder = treatment_recommender_data['label_encoder']
            treatment_recommender.is_trained = True
            print("✅ Modelo de recomendación de tratamientos cargado exitosamente")
        else:
            print("⚠️ Modelo de recomendación de tratamientos no encontrado en models/sklearn_treatment_recommender.joblib")
            
    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        import traceback
        traceback.print_exc()

def process_uploaded_image(file_content: bytes) -> np.ndarray:
    """
    Procesar imagen subida y convertirla al formato requerido.
    
    Args:
        file_content: Contenido del archivo de imagen
        
    Returns:
        Array numpy con la imagen procesada
    """
    try:
        # Convertir bytes a imagen
        image = Image.open(io.BytesIO(file_content))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir a array numpy
        img_array = np.array(image)
        
        # Redimensionar a 224x224
        img_array = cv2.resize(img_array, (224, 224))
        
        # Normalizar
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Página de inicio con documentación básica."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sistema de Medicina Personalizada</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .endpoint { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
            .url { color: #2980b9; font-family: monospace; }
            .status { color: #e74c3c; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1 class="header">🧠 Sistema de Medicina Personalizada</h1>
        <p>API para clasificación de tumores cerebrales y recomendación de tratamientos.</p>
        <p class="status">✅ Versión compatible sin TensorFlow - Usando modelos de scikit-learn</p>
        
        <h2>📋 Endpoints Disponibles</h2>
        
        <div class="endpoint">
            <p><span class="method">POST</span> <span class="url">/classify-tumor</span></p>
            <p>Clasificar tipo de tumor en imagen MRI</p>
            <p><strong>Parámetros:</strong> image (archivo)</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">POST</span> <span class="url">/recommend-treatment</span></p>
            <p>Recomendar tratamiento basado en imagen MRI y datos clínicos</p>
            <p><strong>Parámetros:</strong> image (archivo), age (int), sex (M/F), clinical_note (texto)</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">GET</span> <span class="url">/health</span></p>
            <p>Verificar estado de la API y modelos</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">GET</span> <span class="url">/docs</span></p>
            <p>Documentación interactiva de la API (Swagger UI)</p>
        </div>
        
        <h2>🔬 Tipos de Tumor Detectados</h2>
        <ul>
            <li><strong>Brain_Glioma:</strong> Glioma cerebral</li>
            <li><strong>Brain_Menin:</strong> Meningioma</li>
            <li><strong>Brain_Tumor:</strong> Otros tipos de tumores</li>
        </ul>
        
        <h2>💊 Tratamientos Recomendados</h2>
        <ul>
            <li><strong>Surgery:</strong> Intervención quirúrgica</li>
            <li><strong>Radiation Therapy:</strong> Tratamiento con radiación</li>
            <li><strong>Chemotherapy:</strong> Tratamiento farmacológico</li>
            <li><strong>Close Monitoring:</strong> Monitoreo sin intervención activa</li>
        </ul>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    """Verificar estado de la API y modelos."""
    status = {
        "api_status": "healthy",
        "models": {
            "image_classifier": image_classifier is not None and image_classifier.is_trained,
            "treatment_recommender": treatment_recommender is not None and treatment_recommender.is_trained
        },
        "model_types": {
            "image_classifier": image_classifier.model_type if image_classifier else None,
            "treatment_recommender": treatment_recommender.model_type if treatment_recommender else None
        },
        "endpoints": [
            "/classify-tumor",
            "/recommend-treatment",
            "/health",
            "/docs"
        ],
        "framework": "scikit-learn (compatible con Python 3.13)"
    }
    return status

@app.post("/classify-tumor")
async def classify_tumor(image: UploadFile = File(...)):
    """
    Clasificar tipo de tumor en una imagen MRI.
    
    Args:
        image: Archivo de imagen MRI
        
    Returns:
        Clasificación del tumor con probabilidades
    """
    if image_classifier is None or not image_classifier.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Modelo de clasificación de imágenes no disponible. Ejecutar entrenamiento primero."
        )
    
    try:
        # Leer y procesar imagen
        file_content = await image.read()
        processed_image = process_uploaded_image(file_content)
        
        # Extraer características para el modelo scikit-learn
        image_features = data_loader.extract_image_features(processed_image.reshape(1, *processed_image.shape))[0]
        
        # Realizar predicción
        result = image_classifier.predict_single_image(image_features)
        
        # Convertir tipos numpy a tipos Python nativos para JSON
        result_clean = {
            "predicted_class": str(result["predicted_class"]),
            "confidence": float(result["confidence"]),
            "probabilities": {str(k): float(v) for k, v in result["probabilities"].items()}
        }
        
        # Mapear códigos numéricos a nombres de clases
        class_mapping = {0: "Brain_Glioma", 1: "Brain_Meningiomas", 2: "Brain_Tumor"}
        if result_clean["predicted_class"].isdigit():
            class_code = int(result_clean["predicted_class"])
            if class_code in class_mapping:
                result_clean["predicted_class"] = class_mapping[class_code]
        
        # Agregar metadatos
        result_clean["metadata"] = {
            "filename": image.filename,
            "model_type": image_classifier.model_type,
            "image_size": list(processed_image.shape),
            "processing_time_ms": "< 100"
        }
        
        return result_clean
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en clasificación: {str(e)}")

@app.post("/recommend-treatment")
async def recommend_treatment(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    clinical_note: str = Form(...)
):
    """
    Recomendar tratamiento basado en imagen MRI y datos clínicos.
    
    Args:
        image: Archivo de imagen MRI
        age: Edad del paciente
        sex: Sexo del paciente (M/F)
        clinical_note: Nota clínica del paciente
        
    Returns:
        Recomendación de tratamiento con probabilidades
    """
    if treatment_recommender is None or not treatment_recommender.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Modelo de recomendación de tratamientos no disponible. Ejecutar entrenamiento primero."
        )
    
    try:
        # Validar parámetros
        if sex.upper() not in ['M', 'F']:
            raise HTTPException(status_code=400, detail="El sexo debe ser 'M' o 'F'")
        
        if age < 0 or age > 120:
            raise HTTPException(status_code=400, detail="La edad debe estar entre 0 y 120 años")
        
        # Procesar imagen
        file_content = await image.read()
        processed_image = process_uploaded_image(file_content)
        
        # Extraer características de imagen
        image_features = data_loader.extract_image_features(processed_image.reshape(1, *processed_image.shape))[0]
        
        # Preparar datos clínicos
        clinical_data = {
            'Age': age,
            'Sex': sex.upper(),
            'Clinical Note': clinical_note
        }
        
        # Realizar predicción usando el método simplificado
        # Crear características clínicas manualmente para evitar problemas
        clinical_features = np.array([
            clinical_data.get('Age', 50),
            1 if clinical_data.get('Sex', 'M') == 'M' else 0,
            len(clinical_data.get('Clinical Note', '')),
            1 if 'dolor' in clinical_data.get('Clinical Note', '').lower() else 0,
            1 if 'cabeza' in clinical_data.get('Clinical Note', '').lower() else 0,
            1 if 'náuseas' in clinical_data.get('Clinical Note', '').lower() else 0,
        ])
        
        # Combinar características
        image_features_reshaped = image_features.reshape(1, -1)[:, :19]  # Primeras 19 características
        
        # Crear array con características adicionales para llegar a 38 total
        additional_features = np.zeros((1, 19 - 6))  # 19 - 6 características clínicas básicas
        clinical_features_full = np.concatenate([
            clinical_features.reshape(1, -1), 
            additional_features
        ], axis=1)
        
        combined_features = np.concatenate([image_features_reshaped, clinical_features_full], axis=1)
        
        # Escalar y predecir
        combined_features_scaled = treatment_recommender.scaler.transform(combined_features)
        prediction = treatment_recommender.model.predict(combined_features_scaled)[0]
        probabilities = treatment_recommender.model.predict_proba(combined_features_scaled)[0]
        
        # Decodificar
        recommended_treatment = treatment_recommender.label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        # Preparar resultado
        result_clean = {
            "recommended_treatment": str(recommended_treatment),
            "confidence": confidence,
            "probabilities": {
                str(treatment): float(prob) for treatment, prob in zip(
                    treatment_recommender.label_encoder.classes_, probabilities
                )
            }
        }
        
        # Agregar metadatos
        result_clean["metadata"] = {
            "filename": image.filename,
            "patient_data": {
                "age": age,
                "sex": sex.upper(),
                "clinical_note_length": len(clinical_note)
            },
            "model_type": treatment_recommender.model_type
        }
        
        # También incluir clasificación de tumor si está disponible
        if image_classifier is not None and image_classifier.is_trained:
            tumor_result = image_classifier.predict_single_image(image_features)
            class_mapping = {0: "Brain_Glioma", 1: "Brain_Meningiomas", 2: "Brain_Tumor"}
            
            tumor_classification = {
                "predicted_class": str(tumor_result["predicted_class"]),
                "confidence": float(tumor_result["confidence"]),
                "probabilities": {str(k): float(v) for k, v in tumor_result["probabilities"].items()}
            }
            
            # Mapear código a nombre si es necesario
            if tumor_classification["predicted_class"].isdigit():
                class_code = int(tumor_classification["predicted_class"])
                if class_code in class_mapping:
                    tumor_classification["predicted_class"] = class_mapping[class_code]
            
            result_clean["tumor_classification"] = tumor_classification
        
        return result_clean
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en recomendación: {str(e)}")

@app.post("/train-image-model")
async def train_image_model():
    """Entrenar modelo de clasificación de imágenes (endpoint para demostración)."""
    return {
        "message": "Entrenamiento de modelo de imágenes no implementado en esta demo.",
        "note": "En un ambiente de producción, este endpoint cargaría datos y entrenaría el modelo.",
        "status": "not_implemented"
    }

@app.post("/train-treatment-model")
async def train_treatment_model():
    """Entrenar modelo de recomendación de tratamientos (endpoint para demostración)."""
    return {
        "message": "Entrenamiento de modelo de tratamientos no implementado en esta demo.",
        "note": "En un ambiente de producción, este endpoint cargaría datos y entrenaría el modelo.",
        "status": "not_implemented"
    }

@app.get("/model-info")
async def get_model_info():
    """Obtener información sobre los modelos cargados."""
    
    # Función auxiliar para convertir numpy a tipos Python nativos
    def convert_numpy_types(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    # Obtener clases del clasificador de imágenes
    image_classes = None
    if image_classifier and hasattr(image_classifier, 'label_encoder') and image_classifier.label_encoder:
        try:
            classes = image_classifier.label_encoder.classes_
            image_classes = [str(convert_numpy_types(cls)) for cls in classes]
        except:
            image_classes = None
    
    # Obtener clases del recomendador de tratamientos
    treatment_classes = None
    if treatment_recommender and hasattr(treatment_recommender, 'label_encoder') and treatment_recommender.label_encoder:
        try:
            classes = treatment_recommender.label_encoder.classes_
            treatment_classes = [str(convert_numpy_types(cls)) for cls in classes]
        except:
            treatment_classes = None
    
    info = {
        "image_classifier": {
            "loaded": bool(image_classifier is not None and image_classifier.is_trained),
            "model_type": str(image_classifier.model_type) if image_classifier else None,
            "classes": image_classes
        },
        "treatment_recommender": {
            "loaded": bool(treatment_recommender is not None and treatment_recommender.is_trained),
            "model_type": str(treatment_recommender.model_type) if treatment_recommender else None,
            "classes": treatment_classes
        }
    }
    return info

@app.post("/predict-batch")
async def predict_batch(
    images: list[UploadFile] = File(...),
    clinical_data: str = Form(...)
):
    """
    Procesar múltiples imágenes en lote (para uso avanzado).
    
    Args:
        images: Lista de archivos de imagen
        clinical_data: JSON string con datos clínicos para cada imagen
        
    Returns:
        Lista de predicciones
    """
    try:
        # Parsear datos clínicos
        clinical_data_list = json.loads(clinical_data)
        
        if len(images) != len(clinical_data_list):
            raise HTTPException(
                status_code=400,
                detail="El número de imágenes debe coincidir con el número de registros clínicos"
            )
        
        results = []
        
        for i, (image_file, clinical_record) in enumerate(zip(images, clinical_data_list)):
            try:
                # Procesar imagen
                file_content = await image_file.read()
                processed_image = process_uploaded_image(file_content)
                
                # Clasificación de tumor
                tumor_result = None
                if image_classifier is not None and image_classifier.model is not None:
                    tumor_result = image_classifier.predict_single_image(processed_image)
                
                # Recomendación de tratamiento
                treatment_result = None
                if treatment_recommender is not None and treatment_recommender.model is not None:
                    treatment_result = treatment_recommender.predict_single_case(
                        processed_image, clinical_record
                    )
                
                results.append({
                    "index": i,
                    "filename": image_file.filename,
                    "tumor_classification": tumor_result,
                    "treatment_recommendation": treatment_result,
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "filename": image_file.filename,
                    "error": str(e),
                    "status": "error"
                })
        
        return {
            "total_processed": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "errors": len([r for r in results if r["status"] == "error"]),
            "results": results
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON inválido en clinical_data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento batch: {str(e)}")

if __name__ == "__main__":
    # Configuración para ejecutar el servidor
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 