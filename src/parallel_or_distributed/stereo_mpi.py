"""Versión distribuida con mpi4py: Barrier, scatter de franjas con halos y gather."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from mpi4py import MPI

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.common.stereo_core import (
    cargar_par_estereo, calcular_franja_sad, dividir_franjas_con_halo, evaluar_disparidad
)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-disp", type=int, default=32)
    parser.add_argument("--win", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t_io = 0.0
    left = right = gt = None
    franjas = None
    bytes_enviados = 0

    if rank == 0:
        t0 = perf_counter()
        left, right, gt = cargar_par_estereo(args.data)
        t_io = perf_counter() - t0
        franjas = dividir_franjas_con_halo(left, right, size, args.win)
        bytes_enviados = sum(f[0].nbytes + f[1].nbytes for f in franjas)

    comm.Barrier()
    t_pipe0 = perf_counter() if rank == 0 else 0.0

    # Distribución de franjas + halos
    t_c0 = perf_counter()
    mi_franja = comm.scatter(franjas, root=0)
    t_scatter = perf_counter() - t_c0

    # Cómputo local en cada rank (incluyendo rank 0)
    left_sub, right_sub, y0, y1, in_y0, in_y1 = mi_franja
    disp_local = calcular_franja_sad(
        left_sub, right_sub, y0, y1, in_y0, in_y1, args.max_disp, args.win
    )

    # Recolección de resultados
    t_c1 = perf_counter()
    parciales = comm.gather(disp_local, root=0)
    t_gather = perf_counter() - t_c1

    if rank == 0:
        t_ens0 = perf_counter()
        disp = np.vstack(parciales)
        t_ensamblado = perf_counter() - t_ens0
        t_pipeline = perf_counter() - t_pipe0
        bytes_recibidos = sum(p.nbytes for p in parciales)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(disp).save(args.out)

        metricas = evaluar_disparidad(disp, gt, args.max_disp)
        resumen = {
            "modo": "mpi",
            "procesos": size,
            "resolucion": f"{left.shape[1]}x{left.shape[0]}",
            "max_disp": args.max_disp,
            "win": args.win,
            "t_io_s": t_io,
            "t_comm_s": t_scatter + t_gather,
            "t_ensamblado_s": t_ensamblado,
            "t_pipeline_s": t_pipeline,
            "fps": 1.0 / t_pipeline if t_pipeline > 0 else 0.0,
            "bytes_transferidos": int(bytes_enviados + bytes_recibidos),
            **metricas,
        }
        print(json.dumps(resumen))


if __name__ == "__main__":
    main()