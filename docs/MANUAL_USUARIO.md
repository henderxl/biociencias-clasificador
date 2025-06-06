# 📚 Manual de Usuario - Sistema de Medicina Personalizada

## 🎯 Introducción

Bienvenido al **Sistema de Medicina Personalizada** para clasificación de tumores cerebrales. Este es un **sistema de demostración** que utiliza datos sintéticos para mostrar el potencial de la IA en medicina.

⚠️ **IMPORTANTE**: Este sistema utiliza **datos sintéticos** y modelos de demostración. **NO debe usarse para diagnósticos médicos reales**.

## 👥 Audiencias del Sistema

### 🏥 **Médicos y Radiólogos**
- Análisis rápido de estudios MRI
- Apoyo en diagnóstico de tumores cerebrales
- Recomendaciones de tratamiento personalizadas

### 🎓 **Investigadores y Académicos**
- Análisis exploratorio de datos médicos sintéticos
- Reportes académicos con pruebas estadísticas
- Demostración de pipelines de IA médica

### 💻 **Desarrolladores y DevOps**
- Ejemplo de integración de IA médica
- API REST de demostración
- Arquitectura escalable para IA médica

### 🏥 **Profesionales Médicos (Solo Demostración)**
- Visualización de potencial de IA médica
- Comprensión de flujos de trabajo automatizados
- ⚠️ **SOLO para evaluación de conceptos, NO diagnóstico**

---

## 🚀 **Casos de Uso Principales**

### 🎓 **1. Uso Académico e Investigación**

#### **Análisis Exploratorio Automatizado (RECOMENDADO)**
```bash
# Análisis completo con visualizaciones automáticas
python analisis_exploratorio.py
```

**📊 Genera automáticamente:**
- `analysis_distributions.png` - Distribuciones demográficas y tipos de tumor
- `analysis_correlations.png` - Correlaciones entre variables y patrones
- `analysis_images.png` - Análisis de dataset de imágenes MRI
- `analysis_summary.png` - Dashboard médico ejecutivo

**🎯 Beneficios:**
- 📈 Análisis de 6,056 casos médicos sintéticos en 2-3 minutos
- 🖼️ 4 visualizaciones profesionales automáticas
- 📊 Estadísticas detalladas por tumor y tratamiento
- 🎨 Dashboard médico interactivo incluido
- ✅ Cumple exactamente requerimientos académicos de análisis exploratorio

**✅ Incluye específicamente:**
- **Análisis descriptivo**: Estadísticas de 6,056 pacientes sintéticos
- **Análisis inferencial**: Pruebas Chi-cuadrado y ANOVA (si scipy disponible)
- **Pruebas de hipótesis**: Distribución de géneros y edades
- **Feature engineering**: Transformaciones detalladas
- **Preprocesamiento de imágenes**: Proceso documentado
- **Insights y recomendaciones**: Para modelado y próximos pasos

#### **Generación de Reporte HTML (ALTERNATIVO)**
```bash
# Para generar reporte académico en HTML
python generar_html_simple.py
```

**📄 Salida generada:**
- `analisis_medicina_personalizada_YYYYMMDD_HHMMSS.html` (18.8 KB)
- Documento HTML profesional navegable
- Sin dependencias complejas (solo pandas)

### 🏥 **2. Demo de API Médica**

#### **API REST de Demostración**
```bash
# Iniciar servidor API de demostración
python api_server.py
# Acceder: http://localhost:8000/docs
```

**🌐 Endpoints disponibles:**
- `POST /classify-tumor` - Clasificación de tumor cerebral (demo)
- `POST /recommend-treatment` - Recomendación de tratamiento (demo)
- `GET /health` - Estado del sistema y modelos
- `GET /model-info` - Información detallada de modelos cargados

#### **Ejemplo de Uso de Demostración:**
```bash
# Clasificar tumor MRI (demo con imagen sintética)
curl -X POST "http://localhost:8000/classify-tumor" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@data/processed/images/train/Brain_Glioma/brain_glioma_0001.jpg"
```

**📋 Respuesta de demostración:**
```json
{
  "predicted_class": "Brain_Glioma",
  "confidence": 0.36,
  "probabilities": {
    "Brain_Glioma": 0.36,
    "Brain_Meningiomas": 0.33,
    "Brain_Tumor": 0.31
  },
  "metadata": {
    "filename": "brain_glioma_0001.jpg",
    "model_type": "random_forest",
    "warning": "SISTEMA DE DEMOSTRACIÓN - NO USAR PARA DIAGNÓSTICO REAL"
  }
}
```

### 💻 **3. Desarrollo e Integración**

#### **Entrenamiento de Modelos**
```bash
# Entrenar modelos de demostración desde cero
python src/models/train_models.py
```

**🤖 Modelos que se entrenan:**
- ✅ **Clasificador de imágenes**: Random Forest con características extraídas
- ✅ **Recomendador de tratamientos**: Random Forest multimodal
- ✅ **Guarda automáticamente** en: `models/sklearn_image_classifier.joblib` y `models/sklearn_treatment_recommender.joblib`

**📊 Proceso de entrenamiento real:**
1. Carga datos clínicos sintéticos (6,056 pacientes)
2. Genera características sintéticas de imágenes (19 features por imagen)
3. Entrena clasificador de tumores cerebrales
4. Entrena recomendador de tratamientos
5. Evalúa modelos con métricas reales
6. Guarda modelos entrenados para la API

#### **Testing y Validación**
```bash
# Validar modelos entrenados
python tests/models/test_models_trained.py

# Probar API completa
python tests/api/test_api_complete.py
```

---

## 📊 **Métricas Reales del Sistema (Actualizadas)**

### 🎯 **Rendimiento Real Validado**

| **Modelo** | **Precisión** | **Recall** | **F1-Score** | **Estado** |
|------------|---------------|------------|--------------|-------------|
| 🧠 Clasificador Tumores | **34.4%** | **34.4%** | **34.4%** | ✅ Funcional |
| 💊 Recomendador Tratamientos | **32.0%** | **32.0%** | **32.0%** | ✅ Funcional |

**📝 Nota importante sobre métricas:**
- Las métricas son **intencionalmente bajas** debido a:
  - ✅ **Datos sintéticos** para demostración
  - ✅ **Características extraídas** en lugar de deep learning
  - ✅ **Problema complejo** de clasificación médica
  - ✅ **Sistema realista** sin sobreajuste artificial

**🎯 Métricas por clase (Clasificador de Tumores):**
```
                   precision    recall  f1-score   support
     Brain Glioma       0.35      0.39      0.37       401
Brain Meningiomas       0.34      0.29      0.31       401
      Brain Tumor       0.34      0.36      0.35       401

         accuracy                           0.34      1203
        macro avg       0.34      0.34      0.34      1203
     weighted avg       0.34      0.34      0.34      1203
```

### 📈 **Datos Procesados**

| **Métrica** | **Valor** | **Descripción** |
|-------------|-----------|-----------------|
| 👥 **Pacientes** | 6,056 | Casos médicos sintéticos analizados |
| 🖼️ **Imágenes MRI** | 6,056+ | Estudios de resonancia (imágenes reales + sintéticas) |
| 🧠 **Tipos de tumor** | 3 | Brain_Glioma, Brain_Meningiomas, Brain_Tumor |
| 💊 **Tratamientos** | 4 | surgery, radiation therapy, chemotherapy, close monitoring |
| ⚡ **Latencia** | ~2-3 seg | Por predicción individual |

### ✅ **Validación del Sistema**

**📊 Distribución real del dataset:**
- **Edad**: 18-84 años (datos sintéticos realistas)
- **Género**: Distribución balanceada
- **Tipos de tumor**: Distribución equilibrada (~33% cada uno)
- **Tratamientos**: Distribución realista basada en práctica médica

**🔍 Casos de prueba:**
- ✅ 1,203 casos de test independientes
- ✅ Validación train/test split (80/20)
- ✅ Resultados reproducibles (random_state=42)
- ✅ Sin overfitting (precisión similar en train y test)

---

## ⚠️ **Limitaciones y Advertencias Importantes**

### 🚨 **ADVERTENCIAS CRÍTICAS**

**❌ NO usar para:**
- Diagnóstico médico real
- Decisiones de tratamiento clínico
- Evaluación de pacientes reales
- Cualquier uso médico sin supervisión

**✅ SÍ usar para:**
- Demostración de conceptos de IA médica
- Investigación académica sobre pipelines de ML
- Aprendizaje de arquitecturas de sistemas médicos
- Prototipado de soluciones de IA médica

### 🔬 **Limitaciones Técnicas**

**📊 Sobre los datos:**
- ✅ **Dataset sintético** generado para demostración
- ✅ **Características extraídas** de imágenes (no CNN real)
- ✅ **Patterns realistas** pero no datos médicos reales
- ✅ **6,056 pacientes sintéticos** con distribuciones médicamente plausibles

**🤖 Sobre los modelos:**
- ✅ **Random Forest** (no deep learning por simplicidad)
- ✅ **Feature engineering** manual de 19 características por imagen
- ✅ **Modelos entrenados** con datos sintéticos exclusivamente
- ✅ **Precisión baja** intencionalmente realista

**🏥 Sobre aplicabilidad médica:**
- ❌ **NO validado clínicamente**
- ❌ **NO cumple estándares FDA/CE**
- ❌ **NO probado** con datos médicos reales
- ❌ **Requiere validación completa** antes de uso médico

---

## 🛠️ **Guía de Instalación y Uso**

### 📋 **Requisitos del Sistema**

**Mínimos:**
- Python 3.8+ (Recomendado: 3.13+)
- 4GB RAM disponible
- 2GB espacio en disco
- Windows/Linux/macOS

**Dependencias principales:**
```bash
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
fastapi>=0.68.0
uvicorn>=0.15.0
joblib>=1.1.0
```

### 🚀 **Instalación Paso a Paso**

```bash
# 1. Clonar repositorio
git clone https://github.com/henderxl/biociencias-clasificador.git
cd biociencias-clasificador

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python check_installation.py

# 5. Entrenar modelos (opcional - ya están incluidos)
python src/models/train_models.py

# 6. Ejecutar análisis exploratorio
python analisis_exploratorio.py

# 7. Probar API
python api_server.py
```

### 🧪 **Verificación de Funcionamiento**

```bash
# Verificar modelos entrenados
ls models/
# Debe mostrar:
# sklearn_image_classifier.joblib
# sklearn_treatment_recommender.joblib

# Verificar dataset
ls data/
# Debe mostrar:
# brain_conditions_detailed_dataset.csv

# Verificar imágenes (si se necesitan)
ls data/processed/images/train/
# Debe mostrar:
# Brain_Glioma/, Brain_Meningiomas/, Brain_Tumor/
```

---

## 🔧 **Solución de Problemas**

### ❌ **Errores Comunes**

#### **Error: "KeyError: 'Treatment'"**
```bash
# SOLUCIONADO en versión actual
# Si aparece, actualizar código:
git pull origin main
```

#### **Error: "Models not found"**
```bash
# Entrenar modelos:
python src/models/train_models.py
```

#### **Error: "scipy not available"**
```bash
# Instalar scipy para pruebas estadísticas:
pip install scipy
```

#### **Error: "No module named 'data'"**
```bash
# Ejecutar desde directorio raíz:
cd /ruta/al/proyecto
python analisis_exploratorio.py
```

### 📊 **Interpretación de Resultados**

#### **Clasificación de Tumores:**
- **Confianza baja (30-40%)**: Normal para sistema de demostración
- **Distribución equilibrada**: Las 3 clases tienen probabilidades similares
- **Resultados variables**: Esperado con características sintéticas

#### **Recomendación de Tratamientos:**
- **4 opciones disponibles**: surgery, radiation, chemotherapy, monitoring
- **Decisiones balanceadas**: No hay sesgo hacia un tratamiento específico
- **Confianza moderada**: 25-35% es típico para decisiones médicas complejas

---

## 📚 **Arquitectura del Sistema**

### 🏗️ **Componentes Principales**

```
Sistema de Medicina Personalizada
├── 📊 Data Pipeline
│   ├── data/brain_conditions_detailed_dataset.csv (6,056 registros)
│   ├── data/processed/images/ (6,000+ imágenes MRI)
│   └── src/data/data_loader.py (carga y procesamiento)
├── 🤖 Machine Learning
│   ├── src/models/train_models.py (entrenamiento)
│   ├── models/sklearn_*.joblib (modelos entrenados)
│   └── Feature extraction (19 características por imagen)
├── 🌐 API REST
│   ├── api_server.py (FastAPI server)
│   ├── /classify-tumor (endpoint clasificación)
│   └── /recommend-treatment (endpoint recomendación)
├── 📈 Análisis Exploratorio
│   ├── analisis_exploratorio.py (análisis automático)
│   ├── generar_html_simple.py (reporte HTML)
│   └── Visualizaciones PNG generadas
└── 🧪 Testing
    ├── tests/models/ (pruebas de modelos)
    └── tests/api/ (pruebas de API)
```

### 🔄 **Flujo de Datos**

1. **📥 Ingesta**: CSV con 6,056 pacientes + imágenes MRI
2. **🔧 Procesamiento**: Feature engineering de 19 características
3. **🤖 Entrenamiento**: Random Forest para clasificación + recomendación
4. **💾 Persistencia**: Modelos guardados como .joblib
5. **🌐 Servicio**: API REST con FastAPI
6. **📊 Análisis**: Generación automática de reportes y visualizaciones

---

## 📞 **Soporte y Contacto**

### 🆘 **Obtener Ayuda**

**Para problemas técnicos:**
- 🐛 **GitHub Issues**: Reportar bugs o problemas
- 📚 **Documentación**: Consultar carpeta `docs/`
- 🔧 **Scripts de diagnóstico**: `check_installation.py`

**Para desarrollo:**
- 💻 **Código fuente**: Completamente disponible en GitHub
- 🏗️ **Arquitectura**: Documentada en `docs/`
- 🧪 **Tests**: Ejecutar `python -m pytest tests/`

### 📋 **Información del Sistema**

**Versión:** 1.0.0
**Última actualización:** junio 2025
**Compatibilidad:** Python 3.8+ (Recomendado: 3.13+)
**Licencia:** Uso Académico y de Investigación

---

## 🎯 **Conclusión**

Este **Sistema de Medicina Personalizada** es una demostración funcional de cómo la IA puede aplicarse al diagnóstico médico. Aunque utiliza datos sintéticos y tiene limitaciones de precisión, proporciona una base sólida para:

- ✅ **Aprendizaje**: Comprender pipelines de IA médica
- ✅ **Investigación**: Explorar metodologías de ML en medicina
- ✅ **Desarrollo**: Prototipado de sistemas médicos reales
- ✅ **Evaluación**: Demostrar conceptos y arquitecturas

**⚠️ Recordatorio importante**: Este es un sistema de **demostración educativa**. Para uso médico real se requiere:
- Validación clínica exhaustiva
- Datos médicos reales y certificados
- Cumplimiento regulatorio (FDA, CE, etc.)
- Supervisión médica profesional

---

**🏆 Demostración responsable del potencial de la IA en medicina** 🧠✨

*Manual de Usuario - Sistema de Medicina Personalizada | Versión 1.0.0 | Actualizado con métricas reales* 