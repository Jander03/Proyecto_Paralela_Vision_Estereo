# Datos del Proyecto 5: Vision Estereo Paralela y Distribuida

## Fuente y Licencia
- Referencia principal: Middlebury Stereo Datasets (escena Cones, 2003, D. Scharstein et al.).
- Respaldo reproducible: Si el servidor de Middlebury bloquea la descarga (HTTP 403), data/descargar_datos.py genera un par estereo sintetico determinista (semilla 42) con ground truth exacto en formato compatible (disparidad escalada x4).

## Resoluciones y Transformaciones
1. middlebury_cones/ (Grande): 450 x 375 px en escala de grises (L), con mapa de disparidad de referencia (disp_gt.png).
2. middlebury_cones_small/ (Pequena): 225 x 187 px (reduccion bilineal al 50% y reescalado de disparidad / 2).

## Reproduccion
    python data/descargar_datos.py
