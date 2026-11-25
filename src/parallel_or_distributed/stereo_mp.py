"""Versión paralela en memoria compartida con multiprocessing.Pool y halos."""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.common.stereo_core import (
    cargar_par_estereo, calcular_franja_sad, dividir_franjas_con_halo, evaluar_disparidad
)


def _worker(tarea):
    left_sub, right_sub, y0, y1, in_y0, in_y1, max_disp, win = tarea
    return calcular_franja_sad(left_sub, right_sub, y0, y1, in_y0, in_y1, max_disp, win)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-disp", type=int, default=32)
    parser.add_argument("--win", type=int, default=7)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = perf_counter()
    left, right, gt = cargar_par_estereo(args.data)
    t_io = perf_counter() - t0

    t_pipe0 = perf_counter()
    franjas = dividir_franjas_con_halo(left, right, args.p, args.win)
    bytes_enviados = sum(f[0].nbytes + f[1].nbytes for f in franjas)
    tareas = [(*f, args.max_disp, args.win) for f in franjas]

    with multiprocessing.Pool(processes=args.p) as pool:
        parciales = pool.map(_worker, tareas)

    t_ens0 = perf_counter()
    disp = np.vstack(parciales)
    t_ensamblado = perf_counter() - t_ens0
    t_pipeline = perf_counter() - t_pipe0
    bytes_recibidos = sum(p.nbytes for p in parciales)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(disp).save(args.out)

    metricas = evaluar_disparidad(disp, gt, args.max_disp)
    resumen = {
        "modo": "multiprocessing",
        "procesos": args.p,
        "resolucion": f"{left.shape[1]}x{left.shape[0]}",
        "max_disp": args.max_disp,
        "win": args.win,
        "t_io_s": t_io,
        "t_comm_s": 0.0,
        "t_ensamblado_s": t_ensamblado,
        "t_pipeline_s": t_pipeline,
        "fps": 1.0 / t_pipeline if t_pipeline > 0 else 0.0,
        "bytes_transferidos": int(bytes_enviados + bytes_recibidos),
        **metricas,
    }
    print(json.dumps(resumen))


if __name__ == "__main__":
    main()
