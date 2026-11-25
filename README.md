# Proyecto 5: Vision Estereo Paralela y Distribuida

Implementacion y analisis experimental de correspondencia estereo por ventanas (SAD) en tres modalidades: Secuencial, Multiprocessing (memoria compartida) y MPI (mpi4py, paso explicito de mensajes con manejo de halos).

## Estructura del Repositorio
- Dockerfile
- LICENSE
- README.md
- requirements.txt
- data/ (README.md, descargar_datos.py)
- src/common/stereo_core.py
- src/sequential/stereo_seq.py
- src/parallel_or_distributed/ (stereo_mp.py, stereo_mpi.py)
- experiments/ (comparar_experimentos.py, generar_graficas.py, results/, figures/, report/)

## Instalacion
    python -m pip install -r requirements.txt

## Ejecucion
1. Preparar datos:
    python data/descargar_datos.py
2. Correr suite completa de experimentos (3 repeticiones, medianas y verificacion pixel a pixel):
    python experiments/comparar_experimentos.py --reps 3
3. Generar graficas:
    python experiments/generar_graficas.py

## Ejecucion con Docker
    docker build -t proyecto5-estereo .
    docker run --rm proyecto5-estereo
