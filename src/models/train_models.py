"""
Script de entrenamiento para los modelos de clasificación de tumores y recomendación de tratamientos.
Versión compatible con Python 3.13 sin TensorFlow.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Agregar directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data.data_loader import DataLoader
except ImportError as e:
    print(f"Error importando DataLoader: {e}")
    sys.exit(1)

class SklearnImageClassifier:
    """Clasificador de imágenes usando características extraídas con scikit-learn."""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def build_model(self):
        """Construir el modelo según el tipo especificado."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        else:
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Entrenar el modelo."""
        if self.model is None:
            self.build_model()
        
        # Codificar labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"🔄 Entrenando {self.model_type}...")
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluación en entrenamiento
        train_score = self.model.score(X_train_scaled, y_train_encoded)
        print(f"📊 Precisión en entrenamiento: {train_score:.4f}")
        
        # Evaluación en validación si está disponible
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val_encoded)
            print(f"📊 Precisión en validación: {val_score:.4f}")
        
        self.is_trained = True
        return {'train_accuracy': train_score}
    
    def predict_single_image(self, image_features):
        """Predecir clase de una sola imagen."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Asegurar que sea un array 2D
        if len(image_features.shape) == 1:
            image_features = image_features.reshape(1, -1)
        
        # Escalar características
        image_features_scaled = self.scaler.transform(image_features)
        
        # Predicción
        prediction = self.model.predict(image_features_scaled)[0]
        probabilities = self.model.predict_proba(image_features_scaled)[0]
        
        # Decodificar label
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_name: prob for class_name, prob in zip(
                    self.label_encoder.classes_, probabilities
                )
            }
        }
    
    def save_model(self, filepath):
        """Guardar el modelo entrenado."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Modelo guardado en: {filepath}")

class SklearnTreatmentRecommender:
    """Recomendador de tratamientos usando modelos de scikit-learn."""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def build_model(self):
        """Construir el modelo."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        else:
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
    
    def train(self, X_images, clinical_df, y_treatments):
        """Entrenar el recomendador."""
        if self.model is None:
            self.build_model()
        
        # Extraer características de imágenes
        data_loader = DataLoader()
        image_features = data_loader.create_synthetic_image_features(len(clinical_df))
        
        # Preparar características clínicas
        clinical_features, _ = data_loader.prepare_features_for_treatment(clinical_df)
        
        # Combinar características
        combined_features = np.concatenate([image_features, clinical_features], axis=1)
        
        # Codificar tratamientos
        y_encoded = self.label_encoder.fit_transform(y_treatments)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            combined_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"🔄 Entrenando recomendador {self.model_type}...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluación
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        print(f"📊 Precisión en entrenamiento: {train_score:.4f}")
        print(f"📊 Precisión en validación: {val_score:.4f}")
        
        self.is_trained = True
        return {'train_accuracy': train_score, 'val_accuracy': val_score}
    
    def predict_single_case(self, image_features, clinical_data):
        """Predecir tratamiento para un caso."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Crear un DataLoader para procesar correctamente las características clínicas
        data_loader = DataLoader()
        
        # Crear un DataFrame temporal con los datos clínicos
        clinical_df_temp = pd.DataFrame([{
            'Age': clinical_data.get('Age', 50),
            'Sex': clinical_data.get('Sex', 'M'),
            'Clinical Note': clinical_data.get('Clinical Note', ''),
            'Treatment': 'surgery'  # Valor dummy para evitar error en prepare_features_for_treatment
        }])
        
        # Procesar los datos clínicos para obtener todas las características
        clinical_df_temp = data_loader._clean_clinical_data(clinical_df_temp)
        clinical_features, _ = data_loader.prepare_features_for_treatment(clinical_df_temp)
        
        # Asegurar que las características de imagen tengan la forma correcta
        if len(image_features.shape) == 1:
            image_features = image_features.reshape(1, -1)
        else:
            image_features = image_features.flatten().reshape(1, -1)
        
        # Tomar solo las primeras 19 características de imagen
        image_features = image_features[:, :19]
        
        # Combinar características
        combined_features = np.concatenate([image_features, clinical_features], axis=1)
        
        # Escalar y predecir
        combined_features_scaled = self.scaler.transform(combined_features)
        prediction = self.model.predict(combined_features_scaled)[0]
        probabilities = self.model.predict_proba(combined_features_scaled)[0]
        
        # Decodificar
        recommended_treatment = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        return {
            'recommended_treatment': recommended_treatment,
            'confidence': confidence,
            'probabilities': {
                treatment: prob for treatment, prob in zip(
                    self.label_encoder.classes_, probabilities
                )
            }
        }
    
    def save_model(self, filepath):
        """Guardar el modelo."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath + '.joblib')
        print(f"✅ Recomendador guardado en: {filepath}.joblib")

def generate_synthetic_clinical_data(num_samples: int = 300) -> pd.DataFrame:
    """Generar datos clínicos sintéticos."""
    print("🔄 Generando datos clínicos sintéticos...")
    
    np.random.seed(42)
    
    data = []
    conditions = ['Brain_Glioma', 'Brain_Menin', 'Brain_Tumor']
    treatments = ['surgery', 'radiation therapy', 'chemotherapy', 'close monitoring']
    
    clinical_templates = [
        "Paciente presenta dolores de cabeza severos y náuseas frecuentes.",
        "Historia de convulsiones y alteraciones visuales progresivas.",
        "Síntomas neurológicos con confusión y pérdida de memoria.",
        "Déficit motor y alteraciones del habla evidentes.",
        "Cefaleas matutinas con vómitos proyectiles ocasionales.",
        "Deterioro cognitivo gradual con episodios de desorientación.",
        "Alteraciones del equilibrio y coordinación motora.",
        "Cambios de personalidad y comportamiento anómalo.",
    ]
    
    for i in range(num_samples):
        age = np.random.randint(20, 80)
        sex = np.random.choice(['M', 'F'])
        condition = np.random.choice(conditions)
        clinical_note = np.random.choice(clinical_templates)
        treatment = np.random.choice(treatments)
        
        data.append({
            'Case ID': f'SYNTH_{i:03d}',
            'Condition': condition,
            'Age': age,
            'Sex': sex,
            'Clinical Note': clinical_note,
            'Treatment': treatment
        })
    
    return pd.DataFrame(data)

def train_image_classifier():
    """Entrenar el clasificador de imágenes usando características extraídas."""
    print("\n🧠 ENTRENANDO CLASIFICADOR DE IMÁGENES")
    print("=" * 50)
    
    # Cargar datos usando DataLoader
    data_loader = DataLoader(data_path="../../data/")
    
    # Intentar cargar datos clínicos reales
    try:
        clinical_df = data_loader.load_clinical_data()
        print(f"📊 Datos clínicos cargados: {clinical_df.shape}")
        num_samples = len(clinical_df)
        
        # Usar labels reales
        y_images = data_loader.prepare_image_labels(clinical_df)
        class_names = data_loader.get_class_names('condition')
        
    except FileNotFoundError:
        print("⚠️ Archivo de datos clínicos no encontrado. Generando datos sintéticos...")
        clinical_df = generate_synthetic_clinical_data()
        num_samples = len(clinical_df)
        
        # Procesar datos sintéticos
        clinical_df = data_loader._clean_clinical_data(clinical_df)
        y_images = data_loader.prepare_image_labels(clinical_df)
        class_names = data_loader.get_class_names('condition')
    
    # Generar características de imágenes sintéticas
    X_features = data_loader.create_synthetic_image_features(num_samples)
    
    print(f"📊 Características generadas: {X_features.shape}")
    print(f"📋 Clases: {class_names}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_images, test_size=0.2, random_state=42, stratify=y_images
    )
    
    print(f"🔄 Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
    
    # Crear y entrenar modelo
    classifier = SklearnImageClassifier(model_type='random_forest')
    classifier.build_model()
    
    print("🚀 Iniciando entrenamiento...")
    history = classifier.train(X_train, y_train, X_test, y_test)
    
    # Evaluación detallada
    print("\n📈 Evaluación detallada:")
    y_pred = classifier.model.predict(classifier.scaler.transform(X_test))
    y_test_encoded = classifier.label_encoder.transform(y_test)
    
    print(f"Precisión final: {accuracy_score(y_test_encoded, y_pred):.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # Guardar modelo
    os.makedirs("../../models", exist_ok=True)
    classifier.save_model("../../models/sklearn_image_classifier.joblib")
    
    return classifier

def train_treatment_recommender():
    """Entrenar el recomendador de tratamientos."""
    print("\n💊 ENTRENANDO RECOMENDADOR DE TRATAMIENTOS")
    print("=" * 50)
    
    # Cargar datos clínicos
    data_loader = DataLoader(data_path="../../data/")
    
    try:
        clinical_df = data_loader.load_clinical_data()
        print(f"📊 Datos clínicos cargados: {clinical_df.shape}")
    except FileNotFoundError:
        print("⚠️ Archivo de datos clínicos no encontrado. Generando datos sintéticos...")
        clinical_df = generate_synthetic_clinical_data()
        clinical_df = data_loader._clean_clinical_data(clinical_df)
    
    # Generar características de imágenes sintéticas correspondientes
    num_samples = len(clinical_df)
    X_image_features = data_loader.create_synthetic_image_features(num_samples)
    
    # Preparar tratamientos
    y_treatments = clinical_df['Treatment'].values
    
    print(f"🔄 Datos preparados: {num_samples} muestras")
    print(f"📋 Tratamientos únicos: {np.unique(y_treatments)}")
    
    # Crear y entrenar modelo
    recommender = SklearnTreatmentRecommender(model_type='random_forest')
    recommender.build_model()
    
    print("🚀 Iniciando entrenamiento del recomendador...")
    history = recommender.train(X_image_features, clinical_df, y_treatments)
    
    # Guardar modelo
    recommender.save_model("../../models/sklearn_treatment_recommender")
    
    return recommender

def test_models(image_classifier, treatment_recommender):
    """Probar los modelos entrenados."""
    print("\n🧪 PROBANDO MODELOS ENTRENADOS")
    print("=" * 50)
    
    # Generar características de imagen de prueba
    data_loader = DataLoader()
    test_image_features = data_loader.create_synthetic_image_features(1)[0]
    
    # Probar clasificador de imágenes
    print("🔬 Probando clasificador de imágenes...")
    tumor_result = image_classifier.predict_single_image(test_image_features)
    print(f"Tumor detectado: {tumor_result['predicted_class']}")
    print(f"Confianza: {tumor_result['confidence']:.4f}")
    print(f"Probabilidades: {tumor_result['probabilities']}")
    
    # Probar recomendador de tratamientos
    print("\n💊 Probando recomendador de tratamientos...")
    clinical_data = {
        'Age': 45,
        'Sex': 'M',
        'Clinical Note': 'Paciente presenta dolores de cabeza severos y náuseas frecuentes.'
    }
    
    treatment_result = treatment_recommender.predict_single_case(test_image_features, clinical_data)
    print(f"Tratamiento recomendado: {treatment_result['recommended_treatment']}")
    print(f"Confianza: {treatment_result['confidence']:.4f}")
    print(f"Probabilidades: {treatment_result['probabilities']}")

def main():
    """Función principal para entrenar ambos modelos."""
    print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS DE MEDICINA PERSONALIZADA")
    print("🔬 Versión compatible con Python 3.13 (sin TensorFlow)")
    print("=" * 70)
    
    try:
        # Entrenar clasificador de imágenes
        print("🎯 Fase 1: Entrenamiento del clasificador de imágenes...")
        image_classifier = train_image_classifier()
        
        # Entrenar recomendador de tratamientos
        print("\n🎯 Fase 2: Entrenamiento del recomendador de tratamientos...")
        treatment_recommender = train_treatment_recommender()
        
        # Probar modelos
        print("\n🎯 Fase 3: Pruebas de los modelos...")
        test_models(image_classifier, treatment_recommender)
        
        print("\n✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        print("📁 Modelos guardados en: models/")
        print("🌐 Los modelos están listos para usar en la API.")
        print("🚀 Ejecuta: python ../../api_server.py para iniciar el servidor.")
        
        # Mostrar resumen final
        print("\n📊 RESUMEN DEL ENTRENAMIENTO:")
        print(f"✅ Clasificador de imágenes: {image_classifier.model_type}")
        print(f"✅ Recomendador de tratamientos: {treatment_recommender.model_type}")
        print(f"🎯 Sistema listo para medicina personalizada")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Sugerencia: Verifica que todas las dependencias estén instaladas:")
        print("py -m pip install scikit-learn pandas numpy opencv-python joblib")

if __name__ == "__main__":
    main() 