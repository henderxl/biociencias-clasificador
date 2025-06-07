# 🧠 Sistema de Medicina Personalizada - Clasificación de Tumores Cerebrales

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-red.svg)](LICENSE)

## 🎯 **Descripción del Proyecto**

Sistema de **demostración** de inteligencia artificial para medicina personalizada que combina **análisis de imágenes MRI** con **procesamiento de datos clínicos** para:

- 🧠 **Clasificación automática** de 3 tipos de tumores cerebrales (Glioma, Meningioma, Otros)
- 💊 **Recomendación personalizada** de tratamientos (Cirugía, Radioterapia, Quimioterapia, Seguimiento)
- 🌐 **API REST completa** para integración en sistemas de demostración
- 📊 **Análisis exploratorio automatizado** con visualizaciones médicas
- 📓 **Reporte HTML profesional** con análisis completo y pruebas estadísticas

⚠️ **IMPORTANTE**: Este es un **sistema de demostración** que utiliza datos sintéticos. **NO debe usarse para diagnósticos médicos reales**.

---

## 🚀 **Inicio Rápido - 3 Comandos**

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar reporte HTML completo
python generar_html_simple.py

# 3. Iniciar API de demostración
python api_server.py
```

**🎉 ¡En 30 segundos tienes el sistema funcionando!**

---

## 📋 **Casos de Uso Principales**

### 🎓 **Para Investigación y Académicos:**
```bash
# Análisis exploratorio con 4 visualizaciones
python analisis_exploratorio.py

# Reporte académico completo en HTML
python generar_html_simple.py

# Notebook interactivo Jupyter
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 🌐 **Para Desarrolladores:**
```bash
# API con documentación Swagger automática
python api_server.py
# Acceder: http://localhost:8000/docs
```

### 🧪 **Para Testing y Validación:**
```bash
# Demo funcional completo del sistema
python test_demo_simple.py

# Validar modelos entrenados
python tests/models/test_models_trained.py

# Test completo de API
python tests/api/test_api_complete.py
```

---

## 📊 **Capacidades Técnicas Reales**

| **Métrica** | **Valor** | **Descripción** |
|-------------|-----------|-----------------|
| 🎯 **Precisión** | **33.2%** | Clasificación con datos sintéticos (baseline teórico: 33.33%) |
| ⚡ **Latencia** | **< 3 seg** | Tiempo de respuesta por predicción |
| 📊 **Casos procesados** | **6,056** | Pacientes sintéticos analizados |
| 🖼️ **Imágenes MRI** | **6,056+** | Estudios organizados (reales + sintéticas) |
| 🧠 **Tipos de tumor** | **3** | Glioma, Meningioma, Otros |
| 💊 **Tratamientos** | **4** | Opciones terapéuticas disponibles |

**📝 Nota sobre métricas:** La precisión del 33.2% es **técnicamente correcta** para datos sintéticos sin señal discriminatoria real. Con 3 clases balanceadas, el baseline teórico es 33.33%.

---

## 🗂️ **Estructura del Proyecto**

```
📁 Sistema-Medicina-Personalizada/
├── 📊 data/                           # Datasets médicos
│   ├── brain_conditions_detailed_dataset.csv    # 6,056 historiales sintéticos
│   └── processed/images/                        # Imágenes MRI organizadas
├── 🤖 models/                         # Modelos entrenados
│   ├── sklearn_image_classifier.joblib         # Clasificador Random Forest
│   └── sklearn_treatment_recommender.joblib    # Recomendador multimodal
├── 🔧 src/                           # Código fuente modular
│   ├── data/data_loader.py                     # Carga y validación de datos
│   └── models/train_models.py                  # Entrenamiento de modelos
├── 🌐 API y Análisis/                # Scripts principales
│   ├── api_server.py                           # API FastAPI de demostración
│   ├── analisis_exploratorio.py                # Análisis automatizado
│   └── generar_html_simple.py                  # Reporte HTML académico
├── 🧪 tests/                         # Testing automatizado profesional
│   ├── api/test_api_complete.py                # Tests completos de API
│   └── models/test_models_trained.py           # Validación de modelos
├── 📚 docs/                          # Documentación completa
│   ├── MANUAL_USUARIO.md                       # Manual completo
│   ├── ARQUITECTURA_AZURE.md                   # Despliegue Azure
│   └── DIAGRAMAS_C4_ARQUITECTURA.md            # Diagramas técnicos
└── 📓 notebooks/                     # Análisis interactivo
    └── 01_exploratory_analysis.ipynb           # Jupyter notebook
```

---

## 📈 **Funcionalidades Principales**

### 🧠 **1. Clasificación de Tumores Cerebrales**
- **Entrada**: Imagen MRI + datos demográficos
- **Algoritmo**: Random Forest con 19 características extraídas
- **Salida**: Tipo de tumor (Glioma/Meningioma/Otros) + confianza
- **Precisión**: 33.2% (apropiada para datos sintéticos de demostración)

### 💊 **2. Recomendación Personalizada de Tratamientos**
- **Entrada**: Resultado clasificación + historial clínico + demografía
- **Algoritmo**: Random Forest multimodal (características combinadas)
- **Salida**: Tratamiento recomendado + justificación
- **Opciones**: Cirugía, Radioterapia, Quimioterapia, Seguimiento

### 📊 **3. Análisis Exploratorio Automatizado**
```bash
python analisis_exploratorio.py
```
**Genera automáticamente:**
- `analysis_distributions.png` - Distribuciones demográficas
- `analysis_correlations.png` - Matriz de correlaciones
- `analysis_images.png` - Muestras de imágenes MRI
- `analysis_summary.png` - Dashboard médico ejecutivo

### 📓 **4. Reporte HTML Académico**
```bash
python generar_html_simple.py
```
**Genera reporte profesional que incluye:**
- ✅ Análisis descriptivo e inferencial completo
- ✅ Pruebas de hipótesis (Chi-cuadrado, ANOVA cuando scipy disponible)
- ✅ Feature engineering detallado
- ✅ Entrenamiento y validación de modelos
- ✅ Métricas reales y limitaciones transparentes

### 🌐 **5. API REST de Demostración**
```bash
python api_server.py
# Acceder documentación: http://localhost:8000/docs
```
**Endpoints disponibles:**
- `POST /classify-tumor` - Clasificación de tumores (demo)
- `POST /recommend-treatment` - Recomendación de tratamientos (demo)
- `GET /health` - Estado del sistema
- `GET /model-info` - Información de modelos

---

## 🛠️ **Instalación y Configuración**

### **Requisitos del Sistema**
- 🐍 **Python**: 3.8+ (Recomendado: 3.13+)
- 💾 **RAM**: 4GB mínimo (8GB recomendado)
- 💿 **Espacio**: 2GB para datos y modelos
- 🌐 **Internet**: Para instalación de dependencias

### **Instalación Paso a Paso**

#### **1. Clonar el Repositorio**
```bash
git clone https://github.com/henderxl/biociencias-clasificador.git
cd biociencias-clasificador
```

#### **2. Crear Entorno Virtual (Recomendado)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac  
source venv/bin/activate
```

#### **3. Instalar Dependencias**
```bash
pip install -r requirements.txt
```

#### **4. Verificar Instalación**
```bash
python check_installation.py
```

#### **5. Entrenar Modelos (Opcional - ya incluidos)**
```bash
python src/models/train_models.py
```

#### **6. Ejecutar Demos**
```bash
# Análisis exploratorio
python analisis_exploratorio.py

# API de demostración
python api_server.py

# Reporte HTML
python generar_html_simple.py
```

---

## 🧪 **Testing y Validación**

### **Ejecutar Tests Completos**
```bash
# Tests de modelos
python tests/models/test_models_trained.py

# Tests de API
python tests/api/test_api_complete.py

# Demo funcional
python test_demo_simple.py
```

### **Verificar Funcionamiento**
```bash
# Estado del sistema
python check_installation.py

# Información de modelos
python -c "from src.models.train_models import *; print('Modelos cargados correctamente')"
```

---

## 📊 **Arquitectura Técnica**

### **Tecnologías Utilizadas**
- **ML Framework**: Scikit-learn (Random Forest)
- **API**: FastAPI con documentación automática
- **Procesamiento**: NumPy, Pandas, OpenCV
- **Visualización**: Matplotlib, Seaborn
- **Testing**: Suite de pruebas automatizadas
- **Datos**: CSV + imágenes organizadas

### **Modelos Implementados**
1. **Clasificador de Imágenes**: Random Forest (19 features sintéticas)
2. **Recomendador de Tratamientos**: Random Forest multimodal
3. **Pipeline de Datos**: Carga, procesamiento y validación
4. **API REST**: Endpoints para clasificación y recomendación

---

## ⚠️ **Limitaciones y Consideraciones**

### **🚨 Limitaciones Críticas**
- **Datos sintéticos**: Sistema de demostración únicamente
- **NO validado clínicamente**: Requiere estudios médicos reales
- **Precisión limitada**: Apropiada para datos sin señal real
- **Solo demostración**: NO usar para diagnósticos médicos

### **📋 Para Uso en Producción**
- ✅ Datos médicos reales certificados
- ✅ Validación con especialistas
- ✅ Cumplimiento regulatorio (FDA, CE)
- ✅ Testing exhaustivo con casos reales
- ✅ Explicabilidad médica implementada

---

## 📚 **Documentación Adicional**

| **Archivo** | **Descripción** | **Audiencia** |
|-------------|-----------------|---------------|
| [MANUAL_USUARIO.md](docs/MANUAL_USUARIO.md) | Manual completo del sistema | Usuarios finales |
| [ARQUITECTURA_AZURE.md](docs/ARQUITECTURA_AZURE.md) | Propuesta de despliegue | DevOps/Arquitectos |
| [DIAGRAMAS_C4_ARQUITECTURA.md](docs/DIAGRAMAS_C4_ARQUITECTURA.md) | Diagramas técnicos | Desarrolladores |

---

## 🎯 **Casos de Uso por Audiencia**

### **🎓 Investigadores/Académicos**
- Análisis exploratorio de datos médicos
- Estudio de pipelines de ML en medicina
- Reportes académicos con metodología completa

### **💻 Desarrolladores**
- Ejemplo de sistema médico con IA
- Arquitectura REST API para salud
- Integración de modelos ML en producción

### **🏥 Evaluadores Médicos**
- Demostración de potencial de IA médica
- Comprensión de limitaciones técnicas
- Evaluación de metodologías responsables

---

## 📞 **Soporte y Contribución**

### **🆘 Obtener Ayuda**
- 🐛 **Issues**: Reportar problemas técnicos
- 📚 **Documentación**: Ver carpeta `docs/`
- 🔧 **Diagnóstico**: Ejecutar `check_installation.py`

### **🤝 Contribuir**
- 🍴 Fork del repositorio
- 🔧 Crear feature branch
- ✅ Agregar tests
- 📝 Actualizar documentación
- 🚀 Crear Pull Request

---

## 📜 **Licencia y Ética**

- **Licencia**: Uso académico y de investigación
- **Ética**: Sistema de demostración transparente
- **Responsabilidad**: NO apto para uso médico sin validación
- **Privacidad**: No maneja datos personales reales

---

## 🏆 **Conclusión**

Este proyecto demuestra una **arquitectura completa** para sistemas de medicina personalizada, implementando mejores prácticas de ML, transparencia ética y documentación profesional. 

**Ideal para**: Aprendizaje, investigación, prototipado y demostración de conceptos de IA médica responsable.

**No apto para**: Diagnósticos médicos reales sin validación clínica exhaustiva.

---

*Sistema de Medicina Personalizada | Versión 1.0.0 | Diciembre 2024*  
*Demostración responsable del potencial de la IA en medicina* 🧠✨ 