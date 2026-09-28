# Clasificación de mamografías mediante VGG16

Proyecto de clasificación de imágenes mamográficas mediante técnicas de
aprendizaje profundo y transferencia de aprendizaje, utilizando la arquitectura
VGG16 preentrenada en ImageNet.

El modelo fue desarrollado para la clasificación binaria de mamografías en
**benignas y malignas**, utilizando imágenes provenientes del dataset CBIS-DDSM.

## Metodología

Las imágenes fueron redimensionadas a 150 × 150 píxeles y preprocesadas mediante
`preprocess_input` de VGG16.

Para aumentar la variabilidad del conjunto de entrenamiento se aplicaron técnicas
de data augmentation:

- Rotación de hasta ±10°
- Zoom de 10%
- Flip horizontal

El modelo utiliza la siguiente arquitectura:

VGG16 → Flatten → Dense (256, ReLU) → Dropout (0.5) → Sigmoid

El entrenamiento se realizó en dos etapas:

1. **Transfer Learning:** entrenamiento del clasificador manteniendo congelada
   la base convolucional de VGG16.
2. **Fine-tuning:** ajuste de las últimas cuatro capas de VGG16 utilizando una
   tasa de aprendizaje reducida.

## Resultados

El modelo fue evaluado sobre un conjunto independiente de 699 mamografías.

| Métrica | Resultado |
|---|---:|
| Exactitud | 81.3% |
| Sensibilidad | 87.7% |
| Especificidad | 76.2% |
| Precisión | 74.5% |
| F1-score | 80.5% |

### Matriz de confusión

- Verdaderos negativos (TN): 297
- Falsos positivos (FP): 93
- Falsos negativos (FN): 38
- Verdaderos positivos (TP): 271

## Dataset

Se utilizó **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**, disponible
públicamente a través de The Cancer Imaging Archive (TCIA).
[The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

Debido a las condiciones de distribución y al tamaño del dataset, las imágenes
mamográficas originales no se incluyen en este repositorio.

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- VGG16
- Scikit-learn
- NumPy
- Matplotlib

## Objetivo del proyecto

Evaluar la capacidad de un modelo basado en transferencia de aprendizaje para
distinguir entre mamografías benignas y malignas, explorando su potencial como
herramienta experimental de apoyo al análisis de imágenes mamográficas.

## Limitaciones y trabajo futuro

Los resultados corresponden a una evaluación experimental y no constituyen una
validación para uso clínico.

Como trabajo futuro se propone ampliar y diversificar el conjunto de datos,
optimizar el entrenamiento, realizar validaciones independientes e incorporar
técnicas de interpretabilidad como Grad-CAM.

## Autor

**Cristhoper Farias Osorio**  
Bioingeniería Médica  
Universidad Católica del Maule

## Referencias

- Lee, R. S., et al. (2017). A curated mammography data set for use in
  computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177.

- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for
  large-scale image recognition. *International Conference on Learning
  Representations (ICLR)*.

## Aviso

Este proyecto fue desarrollado con fines académicos y de investigación.
El modelo no está destinado al diagnóstico clínico ni sustituye la evaluación
realizada por profesionales de la salud.
