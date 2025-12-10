# INFORME TÉCNICO - PROYECTO FINAL
## Visión por Computadora III - Maestría en Inteligencia Artificial - UBA

**Autor:** Sevann Radhak Triztan  
**Fecha:** Diciembre de 2025  
**Repositorio:** https://github.com/sevann-radhak/UBA-VPC3.git

---

## 1. INTRODUCCIÓN

### 1.1. Objetivo del proyecto

El objetivo principal de este proyecto es implementar y entrenar un modelo de visión por computadora para clasificación de imágenes utilizando técnicas de fine-tuning sobre un modelo preentrenado. El proyecto demuestra la capacidad de:

- Cargar y preprocesar un dataset de imágenes.
- Realizar fine-tuning de un modelo Vision Transformer (MobileViT).
- Registrar y monitorear métricas durante el entrenamiento usando MLflow.
- Evaluar el modelo en un conjunto de test independiente.
- Comparar métricas entre entrenamiento, validación y test.

### 1.2. Justificación

La clasificación de imágenes es una tarea fundamental en visión por computadora con aplicaciones prácticas en múltiples dominios. Este proyecto utiliza CIFAR-10, un dataset estándar en la comunidad, que permite:

- Validar la implementación con un dataset conocido y bien documentado.
- Comparar resultados con otros trabajos en la literatura.
- Demostrar el proceso completo de desarrollo de un modelo de visión.
- Aplicar buenas prácticas de ML (tracking, versionado, evaluación).

### 1.3. Alcance

**Incluido:**
- Fine-tuning de modelo preentrenado (MobileViT Small).
- Entrenamiento y evaluación con métricas estándar.
- Registro completo en MLflow.
- Código modular y preproducción.
- Inferencia sobre imágenes individuales.

**Limitaciones:**
- Entrenamiento con subset del dataset (10,000 muestras) para optimizar tiempo de ejecución.
- 3 épocas de entrenamiento (suficiente para demostrar el proceso).
- Resolución de imagen reducida (160x160) para acelerar procesamiento.

---

## 2. METODOLOGÍA

### 2.1. Modelo elegido

**Nombre del modelo:** MobileViT Small.
**Fuente:** Hugging Face (apple/mobilevit-small).  
**Arquitectura:** MobileViT es una arquitectura híbrida que combina las ventajas de Vision Transformers (ViT) con eficiencia computacional para dispositivos móviles. Utiliza bloques MobileViT que procesan información local y global de manera eficiente.

**Razones de elección:**
- **Tamaño reducido:** ~5M parámetros, ideal para recursos limitados.
- **Eficiencia:** diseñado para ser rápido en inferencia.
- **Preentrenado:** disponible en ImageNet, permite fine-tuning eficiente.
- **Compatibilidad:** integrado con Hugging Face Transformers.

**Parámetros:**
- Parámetros totales: 4,944,042.
- Parámetros entrenables: 4,944,042 (fine-tuning completo).
- Tamaño del modelo: ~22.5 MB.

### 2.2. Dataset

**Nombre del dataset:** CIFAR-10.  
**Origen:** Hugging Face Datasets.  
**Tamaño completo:**
- Training: 50,000 imágenes.
- Test: 10,000 imágenes.
- Clases: 10 (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión).

**Tamaño utilizado (optimización):**
- Training subset: 10,000 imágenes.
- Validation: 5,000 imágenes (mitad del test original).
- Test: 5,000 imágenes (mitad del test original).

**Características:**
- Resolución original: 32x32 píxeles.
- Resolución procesada: 160x160 píxeles (redimensionado).
- Formato: RGB (3 canales).
- Normalización: Media [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225] (ImageNet).

**División:**
- Training: 10,000 muestras (80% del subset).
- Validation: 5,000 muestras (50% del test original).
- Test: 5,000 muestras (50% del test original).

### 2.3. Preprocesamiento

**Resize:** 160x160 píxeles (optimizado para velocidad).

**Normalización:**
- Media: [0.485, 0.456, 0.406].
- Desviación estándar: [0.229, 0.224, 0.225].
- Valores estándar de ImageNet (compatibilidad con modelo preentrenado).

**Data Augmentation:**
- **Modo rápido (fast_mode):** sin augmentation (optimización de velocidad).
- **Modo normal:** RandomHorizontalFlip (p=0.5) para entrenamiento.

**Transformaciones de validación/test:**
- Solo resize y normalización (sin augmentation).

### 2.4. Configuración de Entrenamiento

**Fine-tuning vs entrenamiento desde cero:**
- **Fine-tuning completo:** todos los parámetros del modelo son entrenables.
- **Razón:** aprovecha conocimiento preentrenado en ImageNet y adapta a CIFAR-10.

**Learning rate:** 0.0001.  
**Razón:** learning rate conservador para fine-tuning, evita destruir conocimiento preentrenado.

**Batch Size:** 128  
**Razón:** balance entre velocidad y estabilidad del gradiente.

**Optimizador:** Adam  
**Razón:** adaptativo, funciona bien con learning rates pequeños.

**Scheduler:** no utilizado.  
**Razón:** entrenamiento corto (3 épocas), no requiere ajuste de LR.

**Número de épocas:** 3  
**Razón:** optimización de tiempo, suficiente para demostrar proceso y obtener métricas razonables.

**Capas congeladas:** ninguna.  
**Razón:** Fine-tuning completo para máxima adaptación al dataset.

**Optimizaciones aplicadas:**
- Mixed Precision (FP16) si GPU disponible: ~2x más rápido.
- torch.compile (GPU): compilación del modelo para inferencia más rápida.
- DataLoader optimizado: num_workers, persistent_workers, prefetch_factor.

---

## 3. IMPLEMENTACIÓN

### 3.1. Estructura del código

```
UBA-VPC3/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml          # Configuración centralizada
├── src/
│   ├── data/
│   │   ├── dataset.py       # CustomDataset y DataLoaders
│   │   └── preprocessing.py # Transformaciones de imágenes
│   ├── models/
│   │   ├── model.py         # VisionModel (wrapper del modelo)
│   │   └── trainer.py       # Clase Trainer para entrenamiento
│   ├── metrics/
│   │   └── metrics.py       # Funciones de cálculo de métricas
│   └── utils/
│       └── helpers.py        # Utilidades (config, device, etc.)
├── scripts/
│   ├── train.py             # Script principal de entrenamiento
│   ├── evaluate.py          # Script de evaluación en test
│   ├── inference.py         # Script de inferencia individual
│   ├── test_config.py       # Verificación de configuración
│   ├── test_dataset.py      # Verificación de dataset
│   └── test_model.py        # Verificación de modelo
├── checkpoints/             # Modelos guardados por época
├── mlruns/                  # MLflow tracking (gitignored)
└── informe/
    └── informe_tecnico.md    # Este documento
```

**Características del código:**
- Modular: separación clara de responsabilidades.
- Preproducción: manejo de errores, logging, configuración externa.
- Reutilizable: componentes independientes y testables.
- Documentado: README completo con instrucciones.

### 3.2. Decisiones técnicas

**Arquitectura del código:**
- Separación en módulos (`data`, `models`, `metrics`, `utils`).
- Configuración centralizada en YAML.
- Scripts independientes para cada tarea (train, evaluate, inference).
- **Razón:** facilita mantenimiento, testing y extensión.

**Congelamiento de capas:**
- No se congelaron capas (fine-tuning completo).
- **Razón:** dataset pequeño (10k muestras), permite adaptación completa sin riesgo de overfitting excesivo.

**Learning rate:**
- 0.0001 (conservador).
- **Razón:** Fine-tuning requiere LR pequeño para no destruir conocimiento preentrenado.

**Métricas seleccionadas:**
- **Entrenamiento:** Loss, Accuracy (train y val).
- **Evaluación:** Accuracy, Precision, Recall, F1-Score.
- **Razón:** métricas estándar para clasificación, permiten comparación train/test.

**Optimizaciones de velocidad:**
- Subset de datos (10k vs 50k).
- Resolución reducida (160 vs 224).
- Fast mode (sin augmentation pesada).
- Batch size grande (128).
- Mixed precision (FP16) en GPU.
- **Razón:** reducir tiempo de ejecución manteniendo calidad demostrable.

### 3.3. Herramientas utilizadas

**MLflow:**
- Versión: 3.7.0.
- Configuración: tracking local (file:./mlruns).
- Registrado:
  - Parámetros: model_name, batch_size, learning_rate, num_epochs, num_classes.
  - Métricas entrenamiento: train_loss, train_accuracy, val_loss, val_accuracy, val_precision, val_recall, val_f1_score (por época).
  - Métricas de test: accuracy, precision, recall, f1_score.
  - Modelo: modelo final guardado como artefacto.

**Hugging face:**
- Transformers: para cargar MobileViT preentrenado.
- Datasets: para cargar CIFAR-10.

**PyTorch:**
- Versión: última estable.
- Características utilizadas:
  - DataLoader optimizado.
  - Mixed Precision (torch.cuda.amp).
  - Model compilation (torch.compile).
  - Checkpointing.

**Otras librerías:**
- torchvision: transformaciones de imágenes.
- scikit-learn: cálculo de métricas (accuracy, precision, recall, f1).
- Pillow: procesamiento de imágenes.
- tqdm: barras de progreso.
- pyyaml: carga de configuración.

---

## 4. RESULTADOS

### 4.1. Métricas de entrenamiento

**Tabla de métricas por época:**

| Época | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 |
|-------|------------|----------------|----------|--------------|---------------|------------|--------|
| 1     | 1.9116     | 51.39%         | 1.3019   | 71.30%       | Calculada     | Calculada  | Calculada |
| 2     | 0.8913     | 80.47%         | 0.5476   | 84.10%       | Calculada     | Calculada  | Calculada |
| 3     | 0.4112     | 89.66%         | 0.3744   | 88.30%       | Calculada     | Calculada  | Calculada |

**Nota:** Precision, Recall y F1-Score se calculan en validación usando las mismas funciones que en test, permitiendo comparación directa entre val y test.

**Evolución de métricas (gráficos en MLflow):**

**Train accuracy:**
- Época 1: ~50% → Época 2: ~80% → Época 3: ~90%.
- Tendencia: crecimiento constante y rápido.

**Train Loss:**
- Época 1: ~1.9 → Época 2: ~0.9 → Época 3: ~0.4.
- Tendencia: reducción rápida y estable.

**Val Accuracy:**
- Época 1: ~71% → Época 2: ~84% → Época 3: ~88%.
- Tendencia: crecimiento consistente, ligeramente por debajo de train.

**Val Loss:**
- Época 1: ~1.3 → Época 2: ~0.55 → Época 3: ~0.37.
- Tendencia: reducción estable, convergiendo.

**Análisis del entrenamiento:**

1. **Convergencia rápida:**
   - El modelo muestra aprendizaje efectivo desde la primera época.
   - Mejora significativa entre épocas 1 y 2.
   - Estabilización en época 3.

2. **Sin overfitting evidente:**
   - Diferencia train/val: ~1.4% (89.66% vs 88.30%).
   - Diferencia mínima indica buena generalización.
   - Val loss sigue reduciéndose, no hay divergencia.

3. **Eficiencia del fine-tuning:**
   - Partiendo de modelo preentrenado, alcanza ~90% accuracy en solo 3 épocas.
   - Demuestra efectividad del fine-tuning vs entrenamiento desde cero.

4. **Comportamiento esperado:**
   - Val accuracy ligeramente menor que train (normal).
   - Ambas métricas mejoran consistentemente.
   - Pérdidas convergen hacia valores bajos.

### 4.2. Métricas de testeo

**Métricas finales en test:**

| Métrica | Valor |
|---------|-------|
| Accuracy | 90.20% |
| Precision | 90.33% |
| Recall | 90.20% |
| F1-Score | 90.19% |

**Comparación train vs Val vs test:**

| Métrica | Train | Val | Test | Diferencia Train-Test |
|---------|-------|-----|------|----------------------|
| Accuracy | 89.66% | 88.30% | 90.20% | +0.54% |
| Precision | - | Calculada | 90.33% | - |
| Recall | - | Calculada | 90.20% | - |
| F1-Score | - | Calculada | 90.19% | - |
| Loss | 0.4112 | 0.3744 | - | - |

**Nota:** las métricas de Precision, Recall y F1-Score se calculan en validación y test usando las mismas funciones, permitiendo comparación directa. Accuracy está disponible en train, val y test para comparación completa.

**Análisis de resultados en test:**

1. **Excelente generalización:**
   - Test accuracy (90.20%) **supera** train accuracy (89.66%).
   - Indica que el modelo no está sobreajustado.
   - Generaliza bien a datos no vistos.

2. **Métricas balanceadas:**
   - Precision (90.33%) ≈ Recall (90.20%) ≈ Accuracy (90.20%).
   - Indica que el modelo no tiene sesgo hacia ninguna clase.
   - F1-Score (90.19%) confirma balance.

3. **Consistencia train/val/test:**
   - Métricas muy similares entre los tres conjuntos.
   - Diferencia mínima indica modelo robusto.
   - No hay evidencia de overfitting.

4. **Calidad de resultados:**
   - >90% en todas las métricas es excelente para CIFAR-10.
   - Considerando subset pequeño y pocas épocas, resultados son muy buenos.
   - Demuestra efectividad del proceso de fine-tuning.

### 4.3. Ejemplos de inferencia

El script `inference.py` permite realizar inferencia sobre imágenes individuales:

```bash
python scripts/inference.py
```

**Funcionalidad:**
- Carga modelo entrenado desde checkpoint.
- Procesa imagen de entrada.
- Muestra clase predicha y confianza.
- Útil para pruebas rápidas y demostraciones.

**Nota:** para ejemplos visuales, ejecutar el script con imágenes de CIFAR-10 y capturar resultados.

### 4.4. Análisis de errores

**Observaciones generales:**
- Con ~90% accuracy, ~10% de las predicciones son incorrectas.
- Métricas balanceadas sugieren que errores están distribuidos entre clases.
- Sin sesgo evidente hacia clases específicas.

**Recomendaciones para análisis futuro:**
- Generar matriz de confusión para identificar clases más confundidas.
- Analizar casos de error específicos.
- Visualizar imágenes mal clasificadas.
- Identificar patrones comunes en errores.

---

## 5. CONCLUSIONES

### 5.1. Análisis de resultados

**¿Qué funcionó bien?**

1. **Fine-tuning efectivo:**
   - El modelo preentrenado se adaptó rápidamente a CIFAR-10.
   - Solo 3 épocas fueron suficientes para alcanzar >90% accuracy.
   - Demuestra la ventaja de usar modelos preentrenados.

2. **Generalización excelente:**
   - Test accuracy supera train accuracy.
   - No hay evidencia de overfitting.
   - Modelo robusto y confiable.

3. **Proceso completo implementado:**
   - Código modular y preproducción.
   - Tracking completo en MLflow.
   - Métricas consistentes entre train/val/test.
   - Evaluación independiente en test.

4. **Optimizaciones exitosas:**
   - Tiempo de entrenamiento reducido significativamente.
   - Manteniendo calidad de resultados.
   - Demuestra eficiencia del proceso.

### 5.2. Limitaciones encontradas

1. **Dataset reducido:**
   - Solo 10,000 muestras de entrenamiento (20% del dataset completo).
   - Limitación: puede no capturar toda la variabilidad.
   - Impacto: menor que el esperado (resultados aún excelentes).

2. **Pocas épocas:**
   - Solo 3 épocas de entrenamiento.
   - Limitación: modelo podría mejorar con más entrenamiento.
   - Impacto: resultados ya son muy buenos, pero hay margen de mejora.

3. **Resolución reducida:**
   - Imágenes procesadas a 160x160 (vs 224x224 estándar).
   - Limitación: pérdida de detalles finos.
   - Impacto: mínimo para CIFAR-10 (imágenes originales son 32x32).

4. **Sin data augmentation:**
   - Fast mode desactivó augmentation pesada.
   - Limitación: menor robustez a variaciones.
   - Impacto: aceptable para demostración, mejorable para producción.

5. **Recursos computacionales:**
   - Entrenamiento en CPU (sin GPU).
   - Limitación: tiempo de ejecución más largo.
   - Impacto: optimizaciones aplicadas mitigaron el problema.

### 5.3. Mejoras futuras

1. **Entrenamiento completo:**
   - Usar dataset completo (50,000 muestras).
   - Entrenar 10-20 épocas
   - **Impacto esperado:** mejora marginal (ya estamos en >90%).

2. **Data augmentation:**
   - Activar augmentation completa (rotaciones, cambios de color, etc.).
   - **Impacto esperado:** mayor robustez, mejor generalización.

3. **Resolución completa:**
   - Usar 224x224 si recursos lo permiten.
   - **Impacto esperado:** mejora en detalles finos.

4. **Análisis detallado:**
   - Generar matriz de confusión.
   - Analizar errores por clase.
   - Visualizar casos de error.
   - **Impacto esperado:** mejor comprensión del modelo.

5. **Hiperparámetros:**
   - Ajustar learning rate (scheduler).
   - Probar diferentes optimizadores.
   - Fine-tuning de capas específicas.
   - **Impacto esperado:** optimización de resultados.

6. **Modelos alternativos:**
   - Probar otros modelos (ViT, EfficientNet, etc.).
   - Comparar resultados.
   - **Impacto esperado:** identificar mejor arquitectura para la tarea.

### 5.4. Aprendizajes del proyecto

1. **Importancia de MLflow:**
   - Tracking sistemático facilita análisis y comparación.
   - Registro de parámetros y métricas es esencial.
   - UI permite visualización rápida de resultados.

2. **Fine-tuning vs entrenamiento desde cero:**
   - Fine-tuning es mucho más eficiente.
   - Modelos preentrenados aceleran desarrollo.
   - Resultados excelentes con pocas épocas.

3. **Métricas consistentes:**
   - Usar mismas métricas en train/val/test permite comparación.
   - Evaluación independiente en test es crucial.
   - Métricas balanceadas indican modelo robusto.

4. **Optimización de tiempo:**
   - Subset de datos y optimizaciones permiten iteración rápida.
   - Importante para desarrollo y pruebas.
   - Balance entre velocidad y calidad.

5. **Código modular:**
   - Estructura modular facilita mantenimiento.
   - Separación de responsabilidades mejora claridad.
   - Configuración externa permite experimentación fácil.

6. **Proceso completo:**
   - Implementar todo el pipeline (train/eval/inference) es valioso.
   - Cada etapa aporta información importante.
   - Evaluación final valida todo el proceso.

---

## 6. REFERENCIAS

1. **MobileViT:**
   - Mehta, S., & Rastegari, M. (2022). "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer". arXiv:2110.02178

2. **CIFAR-10 Dataset:**
   - Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images". University of Toronto

3. **Hugging Face:**
   - Transformers Library: https://huggingface.co/docs/transformers
   - MobileViT Model: https://huggingface.co/apple/mobilevit-small
   - Datasets Library: https://huggingface.co/docs/datasets

4. **MLflow:**
   - Documentación oficial: https://mlflow.org/docs/latest/index.html
   - Tracking API: https://mlflow.org/docs/latest/tracking.html

5. **PyTorch:**
   - Documentación oficial: https://pytorch.org/docs/stable/index.html
   - Mixed Precision Training: https://pytorch.org/docs/stable/amp.html

6. **Vision Transformers:**
   - Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". arXiv:2010.11929

---

## ANEXOS

### Anexo A: configuración completa

**config/config.yaml:**
```yaml
model:
  name: "apple/mobilevit-small"
  num_classes: 10
  freeze_backbone: false
  image_size: 160

training:
  batch_size: 128
  learning_rate: 0.0001
  num_epochs: 3
  device: "cuda"
  use_compile: true
  fast_mode: true

data:
  dataset_name: "cifar10"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  subset_size: 10000

mlflow:
  experiment_name: "vision_project"
  tracking_uri: "file:./mlruns"
```

### Anexo B: métricas detalladas de MLflow

**Run de entrenamiento (valuable-elk-501):**

Parámetros:
- model_name: apple/mobilevit-small
- batch_size: 128
- learning_rate: 0.0001
- num_epochs: 3
- num_classes: 10

Métricas finales:
- train_loss: 0.4112
- train_accuracy: 89.66%
- val_loss: 0.3744
- val_accuracy: 88.30%

**Run de evaluación:**

Métricas:
- accuracy: 0.9020 (90.20%)
- precision: 0.9033 (90.33%)
- recall: 0.9020 (90.20%)
- f1_score: 0.9019 (90.19%)

### Anexo C: comandos de ejecución

**Entrenamiento:**
```bash
python scripts/train.py
```

**Evaluación:**
```bash
python scripts/evaluate.py
```

**Inferencia:**
```bash
python scripts/inference.py
```

**Visualizar MLflow:**
```bash
mlflow ui
# Abrir en navegador: http://localhost:5000
```

---


