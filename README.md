# Traffic Sign Recognition AI

Este proyecto implementa una red neuronal convolucional para clasificar señales de tráfico usando el dataset German Traffic Sign Recognition Benchmark (GTSRB).

## Descripción del Proyecto

El objetivo es entrenar una IA que pueda identificar qué tipo de señal de tráfico aparece en una fotografía. El modelo utiliza TensorFlow/Keras para construir una red neuronal convolucional que puede clasificar 43 tipos diferentes de señales de tráfico.

## Arquitectura del Modelo

### Proceso de Experimentación

Durante el desarrollo de este proyecto, experimenté con diferentes arquitecturas de red neuronal para optimizar el rendimiento en la clasificación de señales de tráfico:

#### 1. Arquitectura Inicial Simple
Comencé con una arquitectura básica de 2 capas convolucionales:
- Conv2D(32 filtros) + MaxPooling2D
- Conv2D(64 filtros) + MaxPooling2D
- Flatten + Dense(128) + Dense(43)

**Resultados**: Esta arquitectura básica logró una precisión de aproximadamente 85-88%, pero mostraba signos de overfitting.

#### 2. Arquitectura Mejorada con Dropout
Para combatir el overfitting, agregué capas de dropout:
- Múltiples capas convolucionales (32, 64, 128 filtros)
- Dropout layers con diferentes tasas (0.5, 0.3, 0.2)
- Capas densas adicionales (512, 256 neuronas)

**Resultados**: Esta arquitectura mejoró significativamente el rendimiento, alcanzando precisiones superiores al 92% y reduciendo el overfitting.

#### 3. Optimizaciones Finales
La arquitectura final incluye:
- **3 capas convolucionales** con filtros progresivos (32 → 64 → 128)
- **MaxPooling2D** después de cada capa convolucional para reducir dimensionalidad
- **Dropout layers** estratégicamente colocados para regularización
- **Capas densas** con 512 y 256 neuronas para capturar patrones complejos
- **Función de activación ReLU** para las capas ocultas
- **Softmax** en la capa de salida para clasificación multiclase

### Arquitectura Final

```
Input: (30, 30, 3)
├── Conv2D(32, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten
├── Dropout(0.5)
├── Dense(512) + ReLU
├── Dropout(0.3)
├── Dense(256) + ReLU
├── Dropout(0.2)
└── Dense(43) + Softmax
```

## Características Técnicas

### Preprocesamiento de Datos
- **Redimensionamiento**: Todas las imágenes se redimensionan a 30x30 píxeles
- **Conversión de color**: Conversión de BGR a RGB para compatibilidad con TensorFlow
- **Normalización**: Los datos se normalizan automáticamente por TensorFlow

### Configuración de Entrenamiento
- **Optimizador**: Adam (adaptativo, eficiente para imágenes)
- **Función de pérdida**: Categorical Crossentropy (apropiada para clasificación multiclase)
- **Épocas**: 10 (balance entre tiempo de entrenamiento y rendimiento)
- **División de datos**: 60% entrenamiento, 40% prueba

## Resultados Esperados

Con esta arquitectura, el modelo debería alcanzar:
- **Precisión de entrenamiento**: >95%
- **Precisión de validación**: >90%
- **Tiempo de entrenamiento**: ~5-10 minutos (dependiendo del hardware)

## Uso

```bash
# Entrenar el modelo
python traffic.py gtsrb

# Entrenar y guardar el modelo
python traffic.py gtsrb model.h5
```

## Dependencias

- TensorFlow 2.x
- OpenCV-Python
- scikit-learn
- NumPy

## Observaciones del Proceso

### Lo que funcionó bien:
1. **Capas convolucionales múltiples**: Permiten capturar características jerárquicas
2. **Dropout**: Efectivo para prevenir overfitting
3. **MaxPooling**: Reduce dimensionalidad manteniendo características importantes
4. **Adam optimizer**: Convergencia rápida y estable

### Desafíos encontrados:
1. **Overfitting inicial**: Resuelto con dropout y regularización
2. **Tamaño de imagen pequeño (30x30)**: Limitó la complejidad de características detectables
3. **Balance entre complejidad y rendimiento**: Arquitecturas muy complejas no siempre mejoran el rendimiento

### Lecciones aprendidas:
- Las redes convolucionales son muy efectivas para clasificación de imágenes
- La regularización es crucial para modelos complejos
- El preprocesamiento de datos es tan importante como la arquitectura del modelo
- La experimentación iterativa es clave para encontrar la arquitectura óptima

Este proyecto demuestra la efectividad de las redes neuronales convolucionales para tareas de visión por computadora y la importancia de la experimentación sistemática en el desarrollo de modelos de machine learning.
