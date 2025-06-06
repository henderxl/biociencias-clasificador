#!/usr/bin/env python3
"""
Script de Análisis Exploratorio de Datos
Sistema de Medicina Personalizada - Clasificación de Tumores Cerebrales

Este script realiza un análisis exploratorio completo sin necesidad de Jupyter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_visualization():
    """Configurar estilo de visualización."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

def print_header(title):
    """Imprimir encabezado decorativo."""
    print("\n" + "="*70)
    print(f"📊 {title}")
    print("="*70)

def print_section(title):
    """Imprimir sección."""
    print(f"\n📋 {title}")
    print("-" * 50)

def load_clinical_data():
    """Cargar y analizar dataset clínico."""
    print_header("ANÁLISIS DEL DATASET CLÍNICO")
    
    data_path = 'data/brain_conditions_detailed_dataset.csv'
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, sep=';')
        print(f"✅ Dataset cargado: {len(df)} registros")
        print(f"📊 Columnas: {list(df.columns)}")
        print(f"📏 Dimensiones: {df.shape}")
    else:
        print("❌ Dataset no encontrado. Creando datos de ejemplo...")
        # Crear datos de ejemplo si no existe el archivo
        np.random.seed(42)
        n_samples = 1000
        
        conditions = ['Brain_Glioma', 'Brain_Menin', 'Brain_Tumor']
        treatments = ['Cirugía', 'Radioterapia', 'Quimioterapia', 'Seguimiento cercano']
        
        df = pd.DataFrame({
            'Case ID': [f'CASE_{i:04d}' for i in range(1, n_samples+1)],
            'Condition': np.random.choice(conditions, n_samples),
            'Age': np.random.randint(20, 80, n_samples),
            'Sex': np.random.choice(['M', 'F'], n_samples),
            'Treatment': np.random.choice(treatments, n_samples)
        })
        print(f"✅ Datos de ejemplo creados: {len(df)} registros")
    
    # Mostrar información básica
    print("\n📋 INFORMACIÓN GENERAL")
    print("-" * 30)
    print(df.info())
    
    print("\n📊 PRIMERAS 5 FILAS")
    print("-" * 30)
    print(df.head())
    
    return df

def analyze_distributions(df):
    """Analizar distribuciones de variables."""
    print_section("ANÁLISIS DE DISTRIBUCIONES")
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Distribución de Variables Categóricas', fontsize=16, fontweight='bold')
    
    # 1. Distribución de Condiciones (Tipos de Tumor)
    condition_counts = df['Condition'].value_counts()
    axes[0,0].pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Distribución de Tipos de Tumor', fontweight='bold')
    
    # 2. Distribución de Género
    sex_counts = df['Sex'].value_counts()
    colors = ['lightblue', 'lightpink']
    axes[0,1].pie(sex_counts.values, labels=['Masculino' if x=='M' else 'Femenino' for x in sex_counts.index], 
                  autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0,1].set_title('Distribución por Género', fontweight='bold')
    
    # 3. Distribución de Tratamientos
    treatment_counts = df['Treatment'].value_counts()
    bars = axes[1,0].bar(range(len(treatment_counts)), treatment_counts.values, 
                        color=sns.color_palette("viridis", len(treatment_counts)))
    axes[1,0].set_xticks(range(len(treatment_counts)))
    axes[1,0].set_xticklabels(treatment_counts.index, rotation=45, ha='right')
    axes[1,0].set_title('Distribución de Tratamientos', fontweight='bold')
    axes[1,0].set_ylabel('Frecuencia')
    
    # Agregar valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Distribución de Edad
    axes[1,1].hist(df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,1].axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, 
                     label=f'Media: {df["Age"].mean():.1f}')
    axes[1,1].axvline(df['Age'].median(), color='green', linestyle='--', linewidth=2, 
                     label=f'Mediana: {df["Age"].median():.1f}')
    axes[1,1].set_title('Distribución de Edad', fontweight='bold')
    axes[1,1].set_xlabel('Edad')
    axes[1,1].set_ylabel('Frecuencia')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('analysis_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Imprimir estadísticas
    print("📊 ESTADÍSTICAS DETALLADAS")
    print("=" * 50)
    print(f"🧠 Tipos de tumor: {condition_counts.to_dict()}")
    print(f"👥 Distribución por género: {sex_counts.to_dict()}")
    print(f"💊 Tratamientos: {treatment_counts.to_dict()}")
    print(f"📅 Edad - Media: {df['Age'].mean():.1f}, Mediana: {df['Age'].median():.1f}, Rango: {df['Age'].min()}-{df['Age'].max()}")
    
    return condition_counts, sex_counts, treatment_counts

def analyze_correlations(df):
    """Analizar correlaciones y patrones."""
    print_section("ANÁLISIS DE CORRELACIONES Y PATRONES")
    
    # Crear visualizaciones de patrones
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔍 Análisis de Patrones y Correlaciones', fontsize=16, fontweight='bold')
    
    # 1. Tratamiento por tipo de tumor
    crosstab_treatment = pd.crosstab(df['Condition'], df['Treatment'])
    crosstab_treatment.plot(kind='bar', ax=axes[0,0], color=sns.color_palette("Set2", 4))
    axes[0,0].set_title('Tratamiento por Tipo de Tumor', fontweight='bold')
    axes[0,0].set_xlabel('Tipo de Tumor')
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].legend(title='Tratamiento', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Edad por tipo de tumor
    df.boxplot(column='Age', by='Condition', ax=axes[0,1])
    axes[0,1].set_title('Distribución de Edad por Tipo de Tumor', fontweight='bold')
    axes[0,1].set_xlabel('Tipo de Tumor')
    axes[0,1].set_ylabel('Edad')
    
    # 3. Género por tipo de tumor
    crosstab_sex = pd.crosstab(df['Condition'], df['Sex'])
    crosstab_sex.plot(kind='bar', ax=axes[1,0], color=['lightblue', 'lightpink'])
    axes[1,0].set_title('Distribución de Género por Tipo de Tumor', fontweight='bold')
    axes[1,0].set_xlabel('Tipo de Tumor')
    axes[1,0].set_ylabel('Frecuencia')
    axes[1,0].legend(['Femenino', 'Masculino'])
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Edad por tratamiento
    df.boxplot(column='Age', by='Treatment', ax=axes[1,1])
    axes[1,1].set_title('Distribución de Edad por Tratamiento', fontweight='bold')
    axes[1,1].set_xlabel('Tratamiento')
    axes[1,1].set_ylabel('Edad')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return crosstab_treatment, crosstab_sex

def analyze_image_directory(base_path):
    """Analizar estructura de directorios de imágenes."""
    stats = {}
    
    if os.path.exists(base_path):
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(base_path, split)
            if os.path.exists(split_path):
                # Contar archivos por clase
                class_counts = {}
                total_files = 0
                
                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    if os.path.isdir(class_path):
                        file_count = len([f for f in os.listdir(class_path) 
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        class_counts[class_name] = file_count
                        total_files += file_count
                
                stats[split] = {
                    'total': total_files,
                    'classes': class_counts
                }
    
    return stats

def analyze_images():
    """Analizar imágenes MRI."""
    print_section("ANÁLISIS DE IMÁGENES MRI")
    
    processed_images_path = 'data/processed/images'
    image_stats = analyze_image_directory(processed_images_path)
    
    if image_stats:
        total_images = sum(split_data['total'] for split_data in image_stats.values())
        print(f"📊 Total de imágenes procesadas: {total_images}")
        
        # Crear visualización de distribución de imágenes
        if total_images > 0:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('📸 Distribución de Imágenes MRI', fontsize=16, fontweight='bold')
            
            # 1. Distribución por split
            splits = list(image_stats.keys())
            split_counts = [image_stats[split]['total'] for split in splits]
            
            axes[0].pie(split_counts, labels=[f'{split.upper()}\n({count})' for split, count in zip(splits, split_counts)],
                       autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Distribución por Split (Train/Val/Test)', fontweight='bold')
            
            # 2. Distribución por clase (usando train set como referencia)
            if 'train' in image_stats and image_stats['train']['classes']:
                class_names = list(image_stats['train']['classes'].keys())
                class_counts = list(image_stats['train']['classes'].values())
                
                colors = sns.color_palette("viridis", len(class_names))
                bars = axes[1].bar(class_names, class_counts, color=colors)
                axes[1].set_title('Distribución por Clase (Training Set)', fontweight='bold')
                axes[1].set_ylabel('Número de Imágenes')
                axes[1].tick_params(axis='x', rotation=45)
                
                # Agregar valores en las barras
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('analysis_images.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Imprimir estadísticas detalladas
            for split, data in image_stats.items():
                print(f"\n🗂️ {split.upper()}:")
                print(f"   Total: {data['total']} imágenes")
                for class_name, count in data['classes'].items():
                    percentage = (count / data['total'] * 100) if data['total'] > 0 else 0
                    print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return image_stats, total_images
    else:
        print("❌ No se encontraron imágenes procesadas")
        print("💡 Ejecutar el script de reorganización: python reorganize_images.py")
        return {}, 0

def generate_insights(df, condition_counts, sex_counts, treatment_counts, image_stats, total_images):
    """Generar insights y recomendaciones."""
    print_section("INSIGHTS Y RECOMENDACIONES")
    
    # Insights del dataset clínico
    print("📊 DATASET CLÍNICO:")
    print("-" * 30)
    
    # Balance de clases
    condition_balance = df['Condition'].value_counts(normalize=True)
    if condition_balance.max() - condition_balance.min() < 0.1:
        print("✅ Dataset balanceado entre tipos de tumor")
    else:
        print("⚠️ Dataset desbalanceado - considerar técnicas de balanceo")
    
    # Distribución de edad
    age_stats = df['Age'].describe()
    print(f"📅 Rango de edad: {age_stats['min']:.0f}-{age_stats['max']:.0f} años")
    print(f"📅 Edad promedio: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} años")
    
    # Distribución de género
    gender_balance = df['Sex'].value_counts(normalize=True)
    if abs(gender_balance['M'] - gender_balance['F']) < 0.1:
        print("✅ Distribución equilibrada por género")
    else:
        print(f"⚠️ Desbalance de género: {gender_balance.to_dict()}")
    
    # Insights de tratamientos
    print("\n💊 TRATAMIENTOS:")
    print("-" * 30)
    most_common_treatment = treatment_counts.index[0]
    print(f"🏆 Tratamiento más común: {most_common_treatment} ({treatment_counts.iloc[0]} casos)")
    
    # Recomendaciones para el modelo
    print("\n🤖 RECOMENDACIONES PARA MODELADO:")
    print("-" * 40)
    recommendations = [
        "✅ Usar validación cruzada estratificada para mantener balance",
        "🔄 Implementar data augmentation para imágenes",
        "📊 Considerar features adicionales como edad en grupos",
        "🎯 Usar métricas balanceadas (F1-score, AUC) además de accuracy",
        "🔍 Implementar explicabilidad (SHAP, LIME) para decisiones clínicas"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    if image_stats and total_images > 0:
        print("\n📸 IMÁGENES MRI:")
        print("-" * 30)
        train_images = image_stats.get('train', {}).get('total', 0)
        
        if train_images > 1000:
            print("✅ Suficientes imágenes para deep learning")
        else:
            print("⚠️ Considerar transfer learning o data augmentation")
        
        train_ratio = train_images / total_images if total_images > 0 else 0
        if 0.6 <= train_ratio <= 0.8:
            print("✅ Split adecuado train/val/test")
        else:
            print(f"⚠️ Revisar split ratio: {train_ratio:.2f} training")
    
    print("\n🎯 PRÓXIMOS PASOS:")
    print("-" * 20)
    next_steps = [
        "🔬 Entrenar modelo de clasificación de imágenes",
        "📝 Desarrollar modelo de procesamiento de texto clínico",
        "🤝 Crear modelo ensemble multimodal",
        "📊 Validar con métricas clínicas relevantes",
        "🚀 Implementar en API para uso hospitalario"
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"{i}. {step}")

def create_summary_dashboard(df, image_stats, total_images):
    """Crear dashboard resumen."""
    print_section("RESUMEN EJECUTIVO")
    
    # Crear resumen visual
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('📋 Dashboard Resumen del Análisis Exploratorio', fontsize=16, fontweight='bold')
    
    gender_balance = df['Sex'].value_counts(normalize=True)
    
    # Crear datos de resumen
    summary_data = {
        'Métrica': [
            'Total Pacientes',
            'Tipos de Tumor',
            'Tratamientos Disponibles',
            'Rango de Edad',
            'Balance de Género',
            'Imágenes Totales'
        ],
        'Valor': [
            f"{len(df):,}",
            str(df['Condition'].nunique()),
            str(df['Treatment'].nunique()),
            f"{df['Age'].min()}-{df['Age'].max()}",
            f"{gender_balance['M']:.1%} M / {gender_balance['F']:.1%} F",
            f"{total_images:,}" if total_images > 0 else 'No disponible'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Crear tabla visual
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, 
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.4, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Estilizar tabla
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:
            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
            cell.set_text_props(weight='normal')
    
    plt.savefig('analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 RESUMEN DE RESULTADOS:")
    print("=" * 50)
    for _, row in summary_df.iterrows():
        print(f"  {row['Métrica']}: {row['Valor']}")

def main():
    """Función principal del análisis exploratorio."""
    print_header("ANÁLISIS EXPLORATORIO DE DATOS - MEDICINA PERSONALIZADA")
    
    # Configurar visualización
    setup_visualization()
    
    # 1. Cargar y analizar datos clínicos
    df = load_clinical_data()
    
    # 2. Analizar distribuciones
    condition_counts, sex_counts, treatment_counts = analyze_distributions(df)
    
    # 3. Analizar correlaciones
    crosstab_treatment, crosstab_sex = analyze_correlations(df)
    
    # 4. Analizar imágenes
    image_stats, total_images = analyze_images()
    
    # 5. Generar insights
    generate_insights(df, condition_counts, sex_counts, treatment_counts, image_stats, total_images)
    
    # 6. Crear dashboard resumen
    create_summary_dashboard(df, image_stats, total_images)
    
    print_header("ANÁLISIS EXPLORATORIO COMPLETADO")
    print("🎉 Análisis completado exitosamente")
    print("✅ Visualizaciones guardadas como archivos PNG")
    print("✅ Insights y recomendaciones generados")
    print("✅ Dashboard resumen creado")
    print("\n📁 Archivos generados:")
    print("   📊 analysis_distributions.png")
    print("   🔍 analysis_correlations.png")
    print("   📸 analysis_images.png")
    print("   📋 analysis_summary.png")
    
    print("\n🚀 Próximos pasos:")
    print("   1. 📊 Revisar las visualizaciones generadas")
    print("   2. 🔬 Ejecutar: py src/models/train_models.py")
    print("   3. 🌐 Probar API: py api_server.py")
    print("   4. 💻 Ejecutar demo: py test_demo_simple.py")

if __name__ == "__main__":
    main() 