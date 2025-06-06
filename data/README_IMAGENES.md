# 🖼️ Imágenes MRI para Medicina Personalizada

## ⚠️ Nota Importante

Este proyecto requiere **6,068 imágenes MRI** para funcionar completamente. Debido al tamaño del dataset (varios GB), las imágenes no están incluidas directamente en el repositorio.

## 📥 Cómo Obtener las Imágenes

### **Opción 1: Dataset Kaggle (Recomendado)**
```bash
# 1. Instalar kaggle CLI
pip install kaggle

# 2. Descargar dataset
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection

# 3. Extraer en data/raw/
unzip brain-mri-images-for-brain-tumor-detection.zip -d data/raw/
```

### **Opción 2: Reorganizar Imágenes Automáticamente**
```bash
# Ejecutar script de reorganización (crea estructura necesaria)
python reorganize_images.py
```

## 📁 Estructura Esperada

```
data/
├── processed/
│   └── images/
│       ├── train/
│       │   ├── Brain_Glioma/     (1,321 imágenes)
│       │   ├── Brain_Meningiomas/ (1,339 imágenes)
│       │   └── Brain_Tumor/       (1,595 imágenes)
│       ├── val/
│       │   ├── Brain_Glioma/     (300 imágenes)
│       │   ├── Brain_Meningiomas/ (306 imágenes)
│       │   └── Brain_Tumor/       (334 imágenes)
│       └── test/
│           ├── Brain_Glioma/     (273 imágenes)
│           ├── Brain_Meningiomas/ (315 imágenes)
│           └── Brain_Tumor/       (285 imágenes)
└── raw/
    └── Brain_Cancer raw MRI data/
        └── Brain_Cancer/
            ├── brain_glioma/
            ├── brain_menin/
            └── brain_tumor/
```

## 🔄 Verificar Instalación

```bash
# Verificar que las imágenes están correctamente organizadas
python tests/test_data_structure.py

# O ejecutar análisis exploratorio
python analisis_exploratorio.py
```

## 📊 Dataset Información

- **Total imágenes**: 6,056
- **Clases**: 3 tipos de tumores cerebrales
- **Formato**: JPG
- **Resolución**: Variable (redimensionadas a 224x224 en procesamiento)
- **Fuente**: Kaggle Brain MRI Images Dataset

## 🚀 Ejecución Sin Imágenes

Si no puedes descargar las imágenes, el proyecto aún funciona con **datos sintéticos**:

```bash
# Entrenar modelos con características sintéticas
python src/models/train_models.py

# Generar análisis
python analisis_exploratorio.py

# Ejecutar API
python api_server.py
```

## 🔗 Enlaces Útiles

- [Dataset Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- [Manual del Usuario](../docs/MANUAL_USUARIO.md)
- [Documentación Técnica](../README.md)

---

**🏥 Proyecto de Medicina Personalizada | Diciembre 2024** 