# Proyecto Final - Visión por Computadora III
## Maestría en Inteligencia Artificial - UBA

### Descripción
Proyecto final de la materia Visión por Computadora III. Implementación de un modelo de visión por computadora con fine-tuning, evaluación y registro de métricas usando MLflow.

### Requisitos del Sistema
- Python 3.8+
- CUDA-capable GPU (recomendado)
- 8GB+ RAM

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/sevann-radhak/UBA-VPC3.git
cd UBA-VPC3
```

2. Crear entorno virtual:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Estructura del Proyecto

```
UBA-VPC3/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── trainer.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── notebooks/
│   └── exploration.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── test_config.py
│   └── test_dataset.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
└── informe/
    └── informe_tecnico.pdf
```

### Uso

#### Verificar Configuración
```bash
python scripts/test_config.py
```

#### Verificar Dataset
```bash
python scripts/test_dataset.py
```

#### Entrenamiento
```bash
python scripts/train.py
```

#### Evaluación
```bash
python scripts/evaluate.py
```

#### Inferencia
```bash
python scripts/inference.py
```

### Visualización de Métricas

Para ver las métricas registradas en MLflow:
```bash
mlflow ui
```

Luego abrir en el navegador: `http://localhost:5000`

### Autores
Sevann Radhak Triztan
sevann.radhak@gmail.com

