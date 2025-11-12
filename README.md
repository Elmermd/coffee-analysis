
# Coffee Survey Analysis

Análisis exploratorio de datos sobre hábitos de consumo de café utilizando Python.

## Acerca del Proyecto

Análisis de patrones de consumo de café en diferentes grupos demográficos, enfocado en diferencias por edad y género.

**Dataset:** Coffee Survey (4,042 encuestados)  
**Fuente:** Data Science Essentials with Python - CISCO Networking Academy

## Hallazgos Principales

### Consumo por Género y Edad

**La brecha de género varía 2.6x entre generaciones**

| Grupo de Edad | Hombres | Mujeres | Diferencia |
|--------------|---------|---------|------------|
| **Gen X (45-64)** | 2.25 | 1.65 | **+0.60**  |
| Older Millennials (35-44) | 1.97 | 1.40 | +0.57 |
| Gen Z (<25) | 1.56 | 1.09 | +0.47 |
| Young Millennials (25-34) | 1.76 | 1.30 | +0.47 |
| Boomers+ (65+) | 2.23 | 2.00 | +0.23 |

**Insights:**
- Gen X muestra la mayor brecha de género (36% más en hombres)
- Boomers+ tiene la menor diferencia (11%)
- Los hombres consumen más café en todos los grupos de edad


## Tecnologías

- Python 3.8+
- pandas, numpy
- plotly
- jupyter

## Metodología

1. Limpieza de datos (imputación, codificación)
2. Análisis estratificado por edad
3. Visualización interactiva con Plotly


##  Agradecimientos

CISCO Networking Academy - Data Science Essentials with Python
