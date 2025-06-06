"""
Módulo para carga y procesamiento de datos del proyecto de medicina personalizada.
Maneja tanto los datos clínicos como las imágenes MRI.
"""

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataLoader:
    """Clase para cargar y procesar datos clínicos e imágenes MRI."""
    
    def __init__(self, data_path: str = "data/", img_size: Tuple[int, int] = (224, 224)):
        """
        Inicializar el cargador de datos.
        
        Args:
            data_path: Ruta al directorio de datos
            img_size: Tamaño al que redimensionar las imágenes (height, width)
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.label_encoder_condition = LabelEncoder()
        self.label_encoder_treatment = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_clinical_data(self, filename: str = "brain_conditions_detailed_dataset.csv") -> pd.DataFrame:
        """
        Cargar datos clínicos desde CSV.
        
        Args:
            filename: Nombre del archivo CSV
            
        Returns:
            DataFrame con los datos clínicos
        """
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
            
        # Usar punto y coma como delimitador (formato del archivo)
        df = pd.read_csv(file_path, delimiter=';')
        
        # Limpiar y procesar datos
        df = self._clean_clinical_data(df)
        
        return df
    
    def _clean_clinical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar y procesar datos clínicos.
        
        Args:
            df: DataFrame con datos clínicos
            
        Returns:
            DataFrame procesado
        """
        # Eliminar filas con valores nulos
        df = df.dropna()
        
        # Convertir sexo a numérico (M=1, F=0)
        df['Sex_Numeric'] = df['Sex'].map({'M': 1, 'F': 0})
        
        # Procesar notas clínicas (longitud como feature)
        df['Clinical_Note_Length'] = df['Clinical Note'].str.len()
        
        # Extraer keywords importantes de las notas clínicas
        keywords = [
            'dolor', 'cabeza', 'convulsiones', 'crisis', 'náuseas', 'vómitos',
            'confusión', 'memoria', 'visual', 'auditivo', 'motor', 'cognitivo',
            'epiléptico', 'tumor', 'presión', 'síntomas'
        ]
        
        for keyword in keywords:
            df[f'keyword_{keyword}'] = df['Clinical Note'].str.lower().str.contains(keyword).astype(int)
        
        return df
    
    def load_single_image(self, image_path: str) -> np.ndarray:
        """
        Cargar y procesar una sola imagen MRI.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Array numpy con la imagen procesada
        """
        try:
            # Cargar imagen
            img = cv2.imread(image_path)
            if img is None:
                # Intentar con PIL si OpenCV falla
                img = Image.open(image_path)
                img = np.array(img)
            
            # Convertir a RGB si es necesario
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar
            img = cv2.resize(img, self.img_size)
            
            # Normalizar
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error cargando imagen {image_path}: {e}")
            return np.zeros((*self.img_size, 3), dtype=np.float32)
    
    def load_images_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Cargar un lote de imágenes.
        
        Args:
            image_paths: Lista de rutas a las imágenes
            
        Returns:
            Array numpy con todas las imágenes
        """
        images = []
        for path in image_paths:
            img = self.load_single_image(path)
            images.append(img)
        
        return np.array(images)
    
    def prepare_features_for_treatment(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preparar features para el modelo de recomendación de tratamiento.
        
        Args:
            df: DataFrame con datos clínicos
            
        Returns:
            Tupla con features y labels
        """
        # Seleccionar features numéricas
        feature_columns = ['Age', 'Sex_Numeric', 'Clinical_Note_Length']
        
        # Agregar features de keywords
        keyword_columns = [col for col in df.columns if col.startswith('keyword_')]
        feature_columns.extend(keyword_columns)
        
        X = df[feature_columns].values
        
        # Escalar features
        X = self.scaler.fit_transform(X)
        
        # Preparar labels
        y = self.label_encoder_treatment.fit_transform(df['Treatment'].values)
        
        return X, y
    
    def prepare_image_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preparar labels para clasificación de imágenes.
        
        Args:
            df: DataFrame con datos clínicos
            
        Returns:
            Array con labels codificadas
        """
        return self.label_encoder_condition.fit_transform(df['Condition'].values)
    
    def extract_image_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extraer características básicas de las imágenes para modelos no-CNN.
        
        Args:
            images: Array de imágenes (N, H, W, C)
            
        Returns:
            Array de características extraídas
        """
        features = []
        
        for img in images:
            # Convertir a escala de grises para análisis
            if len(img.shape) == 3:
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (img * 255).astype(np.uint8)
            
            # Extraer estadísticas básicas
            feature_vector = [
                np.mean(gray),           # Media de intensidad
                np.std(gray),            # Desviación estándar
                np.median(gray),         # Mediana
                np.max(gray),            # Valor máximo
                np.min(gray),            # Valor mínimo
                np.percentile(gray, 25), # Percentil 25
                np.percentile(gray, 75), # Percentil 75
            ]
            
            # Extraer características de textura usando gradientes
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            feature_vector.extend([
                np.mean(magnitude),      # Media del gradiente
                np.std(magnitude),       # Std del gradiente
            ])
            
            # Características de distribución de píxeles
            hist, _ = np.histogram(gray, bins=10, range=(0, 255))
            hist = hist / np.sum(hist)  # Normalizar histograma
            feature_vector.extend(hist.tolist())
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_train_test_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Crear división de entrenamiento y prueba.
        
        Args:
            X: Features
            y: Labels
            test_size: Proporción para prueba
            random_state: Semilla aleatoria
            
        Returns:
            Tupla con X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def get_class_names(self, label_type: str = 'condition') -> List[str]:
        """
        Obtener nombres de las clases.
        
        Args:
            label_type: Tipo de label ('condition' o 'treatment')
            
        Returns:
            Lista con nombres de las clases
        """
        if label_type == 'condition':
            return list(self.label_encoder_condition.classes_)
        elif label_type == 'treatment':
            return list(self.label_encoder_treatment.classes_)
        else:
            raise ValueError("label_type debe ser 'condition' o 'treatment'")
    
    def get_label_encoder(self, label_type: str):
        """
        Obtener el encoder de labels correspondiente.
        
        Args:
            label_type: Tipo de label ('condition' o 'treatment')
            
        Returns:
            LabelEncoder correspondiente
        """
        if label_type == 'condition':
            return self.label_encoder_condition
        elif label_type == 'treatment':
            return self.label_encoder_treatment
        else:
            raise ValueError("label_type debe ser 'condition' o 'treatment'")
    
    # Método para compatibilidad con sistemas sin imágenes reales
    def create_synthetic_image_features(self, num_samples: int) -> np.ndarray:
        """
        Crear características sintéticas de imágenes para testing.
        
        Args:
            num_samples: Número de muestras a generar
            
        Returns:
            Array de características sintéticas
        """
        np.random.seed(42)
        # Generar 19 características por imagen (como las extraídas realmente)
        features = np.random.randn(num_samples, 19)
        
        # Hacer que las características sean más realistas
        features[:, 0] = np.random.uniform(50, 200, num_samples)  # Media de intensidad
        features[:, 1] = np.random.uniform(10, 50, num_samples)   # Std
        features[:, 2] = np.random.uniform(40, 180, num_samples)  # Mediana
        
        return features

class DataValidator:
    """Clase para validar la calidad y consistencia de los datos."""
    
    @staticmethod
    def validate_clinical_data(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validar datos clínicos.
        
        Args:
            df: DataFrame con datos clínicos
            
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicated_records': df.duplicated().sum(),
            'age_range': {'min': df['Age'].min(), 'max': df['Age'].max()},
            'sex_distribution': df['Sex'].value_counts().to_dict(),
            'condition_distribution': df['Condition'].value_counts().to_dict(),
            'treatment_distribution': df['Treatment'].value_counts().to_dict()
        }
        
        return validation_results
    
    @staticmethod
    def validate_image_data(image_paths: List[str]) -> Dict[str, any]:
        """
        Validar datos de imágenes.
        
        Args:
            image_paths: Lista de rutas a las imágenes
            
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            'total_images': len(image_paths),
            'existing_files': 0,
            'missing_files': 0,
            'corrupted_files': 0,
            'valid_formats': 0
        }
        
        for path in image_paths:
            if os.path.exists(path):
                validation_results['existing_files'] += 1
                try:
                    img = Image.open(path)
                    img.verify()
                    validation_results['valid_formats'] += 1
                except:
                    validation_results['corrupted_files'] += 1
            else:
                validation_results['missing_files'] += 1
        
        return validation_results 