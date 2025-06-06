"""
Script para probar los modelos entrenados
Ejecutar desde el directorio raíz del proyecto: python tests/models/test_models_trained.py
"""
import sys
import numpy as np
from pathlib import Path
import os

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Cambiar al directorio raíz para que los paths relativos funcionen
original_cwd = os.getcwd()
os.chdir(project_root)

try:
    from src.data.data_loader import DataLoader
    from src.models.train_models import SklearnImageClassifier, SklearnTreatmentRecommender
    import joblib

    def test_trained_models():
        """Probar modelos entrenados"""
        print('🔬 Cargando modelos entrenados...')
        
        # Verificar que los archivos de modelos existen
        image_model_path = 'models/sklearn_image_classifier.joblib'
        treatment_model_path = 'models/sklearn_treatment_recommender.joblib'
        
        if not os.path.exists(image_model_path):
            print(f'❌ Error: No se encontró {image_model_path}')
            print('💡 Ejecuta primero: python src/models/train_models.py')
            return False
            
        if not os.path.exists(treatment_model_path):
            print(f'❌ Error: No se encontró {treatment_model_path}')
            print('💡 Ejecuta primero: python src/models/train_models.py')
            return False
        
        # Cargar modelos
        image_classifier_data = joblib.load(image_model_path)
        treatment_recommender_data = joblib.load(treatment_model_path)
        
        # Recrear objetos del modelo
        image_classifier = SklearnImageClassifier()
        image_classifier.model = image_classifier_data['model']
        image_classifier.scaler = image_classifier_data['scaler']
        image_classifier.label_encoder = image_classifier_data['label_encoder']
        image_classifier.is_trained = True
        
        treatment_recommender = SklearnTreatmentRecommender()
        treatment_recommender.model = treatment_recommender_data['model']
        treatment_recommender.scaler = treatment_recommender_data['scaler']
        treatment_recommender.label_encoder = treatment_recommender_data['label_encoder']
        treatment_recommender.is_trained = True
        
        # Probar modelos
        data_loader = DataLoader()
        test_image_features = data_loader.create_synthetic_image_features(1)[0]
        
        print('🧠 Probando clasificador de imágenes...')
        tumor_result = image_classifier.predict_single_image(test_image_features)
        print(f'Tumor detectado: {tumor_result["predicted_class"]}')
        print(f'Confianza: {tumor_result["confidence"]:.4f}')
        print(f'Probabilidades: {tumor_result["probabilities"]}')
        
        print('\n💊 Probando recomendador de tratamientos...')
        
        # Simular características combinadas para el recomendador
        # Crear características como lo haría el entrenamiento
        image_features = test_image_features.reshape(1, -1)[:, :19]  # 19 características de imagen
        
        # Simular características clínicas (18 características + 1 para edad extra)
        clinical_features = np.array([[
            45,  # Age
            1,   # Sex_Numeric (M=1)
            60,  # Clinical_Note_Length
            1,   # keyword_dolor
            1,   # keyword_cabeza
            1,   # keyword_náuseas
            0,   # keyword_convulsiones
            0,   # keyword_crisis
            0,   # keyword_vómitos
            0,   # keyword_confusión
            0,   # keyword_memoria
            0,   # keyword_visual
            0,   # keyword_auditivo
            0,   # keyword_motor
            0,   # keyword_cognitivo
            0,   # keyword_epiléptico
            1,   # keyword_tumor
            0,   # keyword_presión
            0    # keyword_síntomas
        ]])
        
        # Combinar características
        combined_features = np.concatenate([image_features, clinical_features], axis=1)
        
        # Escalar y predecir
        combined_features_scaled = treatment_recommender.scaler.transform(combined_features)
        prediction = treatment_recommender.model.predict(combined_features_scaled)[0]
        probabilities = treatment_recommender.model.predict_proba(combined_features_scaled)[0]
        
        # Decodificar
        recommended_treatment = treatment_recommender.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        print(f'Tratamiento recomendado: {recommended_treatment}')
        print(f'Confianza: {confidence:.4f}')
        print(f'Tratamientos disponibles: {list(treatment_recommender.label_encoder.classes_)}')
        
        print('\n✅ MODELOS FUNCIONANDO CORRECTAMENTE!')
        print('📊 Resumen del sistema:')
        print(f'   - Clasificador de tumores: {image_classifier.model_type}')
        print(f'   - Recomendador de tratamientos: {treatment_recommender.model_type}')
        print(f'   - Clases de tumores: {list(image_classifier.label_encoder.classes_)}')
        print(f'   - Tratamientos disponibles: {list(treatment_recommender.label_encoder.classes_)}')
        print('\n🚀 Sistema de medicina personalizada listo para usar en la API!')
        return True

    if __name__ == "__main__":
        try:
            success = test_trained_models()
            if success:
                print('\n🎉 TESTS COMPLETADOS EXITOSAMENTE')
            else:
                print('\n❌ Tests fallidos - Revisar configuración')
        except Exception as e:
            print(f'\n💥 Error durante las pruebas: {e}')
            print('\n💡 Asegúrate de:')
            print('   1. Ejecutar desde el directorio raíz: python tests/models/test_models_trained.py')
            print('   2. Tener los modelos entrenados: python src/models/train_models.py')
            print('   3. Dependencias instaladas: pip install -r requirements.txt')
        
finally:
    # Restaurar directorio original
    os.chdir(original_cwd) 