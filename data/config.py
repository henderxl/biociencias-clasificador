# Configuración de datos para el sistema de medicina personalizada
# Generado automáticamente por reorganize_images.py

# Rutas de datos procesados
PROCESSED_IMAGES_PATH = "data/processed/images"
TRAIN_PATH = "data/processed/images/train"
VAL_PATH = "data/processed/images/val" 
TEST_PATH = "data/processed/images/test"

# Clases de tumores
TUMOR_CLASSES = ["Brain_Glioma", "Brain_Menin", "Brain_Tumor"]

# Configuración de entrenamiento
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Configuración de augmentación
USE_AUGMENTATION = True
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True
