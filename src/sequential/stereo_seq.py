"""Línea base secuencial para correspondencia estéreo (1 proceso)."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.common.stereo_core import cargar_par_estereo, calcular_franja_sad, evaluar_disparidad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-disp", type=int, default=32)
    parser.add_argument("--win", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = perf_counter()
    left, right, gt = cargar_par_estereo(args.data)
    t_io = perf_counter() - t0

    t1 = perf_counter()
    h = left.shape[0]
    disp = calcular_franja_sad(left, right, 0, h, 0, h, args.max_disp, args.win)
    t_pipeline = perf_counter() - t1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(disp).save(args.out)

    metricas = evaluar_disparidad(disp, gt, args.max_disp)
    resumen = {
        "modo": "secuencial",
        "procesos": 1,
        "resolucion": f"{left.shape[1]}x{left.shape[0]}",
        "max_disp": args.max_disp,
        "win": args.win,
        "t_io_s": t_io,
        "t_comm_s": 0.0,
        "t_ensamblado_s": 0.0,
        "t_pipeline_s": t_pipeline,
        "fps": 1.0 / t_pipeline if t_pipeline > 0 else 0.0,
        "bytes_transferidos": 0,
        **metricas,
    }
    print(json.dumps(resumen))


if __name__ == "__main__":
    main()