"""Automatiza los experimentos del Proyecto 5, verifica correctitud y guarda CSVs."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
MAPS_DIR = RESULTS_DIR / "mapas"

DATASETS = [
    ("small", ROOT / "data" / "middlebury_cones_small"),
    ("large", ROOT / "data" / "middlebury_cones"),
]
MAX_DISPS = [32, 64]
WIN_SIZE = 7
PROCS = [1, 2, 4]


def mismas_imagenes(ruta_a: Path, ruta_b: Path) -> bool:
    with Image.open(ruta_a) as a, Image.open(ruta_b) as b:
        return a.size == b.size and a.tobytes() == b.tobytes()


def ejecutar_comando(cmd: list[str]):
    t0 = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_lanzamiento = perf_counter() - t0
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"Fallo al ejecutar: {' '.join(cmd)}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    datos = json.loads(proc.stdout.strip().splitlines()[-1])
    datos["t_lanzamiento_s"] = t_lanzamiento
    return datos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3, help="Repeticiones por escenario")
    parser.add_argument("--sin-mpi", action="store_true", help="Omitir pruebas MPI")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    seq_script = str(ROOT / "src" / "sequential" / "stereo_seq.py")
    mp_script = str(ROOT / "src" / "parallel_or_distributed" / "stereo_mp.py")
    mpi_script = str(ROOT / "src" / "parallel_or_distributed" / "stereo_mpi.py")

    detalle = []
    resumen = []

    for ds_name, ds_path in DATASETS:
        for max_disp in MAX_DISPS:
            print(f"\n=== Escenario: {ds_name} | max_disp={max_disp} | win={WIN_SIZE} ===")
            ref_img = MAPS_DIR / f"{ds_name}_d{max_disp}_secuencial_p1.png"

            configuraciones = [("secuencial", 1)]
            configuraciones += [("multiprocessing", p) for p in PROCS]
            if not args.sin_mpi:
                configuraciones += [("mpi", p) for p in PROCS]

            t_base_pipe = None
            t_base_lanz = None

            for modo, p in configuraciones:
                out_img = MAPS_DIR / f"{ds_name}_d{max_disp}_{modo}_p{p}.png"

                if modo == "secuencial":
                    cmd = [py, seq_script, "--data", str(ds_path),
                           "--max-disp", str(max_disp), "--win", str(WIN_SIZE),
                           "--out", str(out_img)]
                elif modo == "multiprocessing":
                    cmd = [py, mp_script, "--data", str(ds_path),
                           "--max-disp", str(max_disp), "--win", str(WIN_SIZE),
                           "--p", str(p), "--out", str(out_img)]
                else:
                    cmd = ["mpiexec", "-n", str(p), py, mpi_script,
                           "--data", str(ds_path), "--max-disp", str(max_disp),
                           "--win", str(WIN_SIZE), "--out", str(out_img)]

                corridas = []
                for rep in range(1, args.reps + 1):
                    r = ejecutar_comando(cmd)
                    correcto = mismas_imagenes(ref_img, out_img)
                    r.update({
                        "dataset": ds_name,
                        "rep": rep,
                        "salida_identica": correcto,
                        "mapa_path": str(out_img),
                    })
                    corridas.append(r)
                    detalle.append(r)

                med_pipe = median(c["t_pipeline_s"] for c in corridas)
                med_lanz = median(c["t_lanzamiento_s"] for c in corridas)
                med_comm = median(c["t_comm_s"] for c in corridas)
                med_ens = median(c["t_ensamblado_s"] for c in corridas)
                med_io = median(c["t_io_s"] for c in corridas)

                if modo == "secuencial":
                    t_base_pipe = med_pipe
                    t_base_lanz = med_lanz

                sp_pipe = t_base_pipe / med_pipe
                ef_pipe = sp_pipe / p
                sp_lanz = t_base_lanz / med_lanz
                ef_lanz = sp_lanz / p
                todas_ok = all(c["salida_identica"] for c in corridas)

                fila_res = {
                    "dataset": ds_name,
                    "resolucion": corridas[0]["resolucion"],
                    "max_disp": max_disp,
                    "win": WIN_SIZE,
                    "modo": modo,
                    "procesos": p,
                    "t_pipeline_s": round(med_pipe, 6),
                    "t_lanzamiento_s": round(med_lanz, 6),
                    "t_comm_s": round(med_comm, 6),
                    "t_ensamblado_s": round(med_ens, 6),
                    "t_io_s": round(med_io, 6),
                    "fps_pipeline": round(1.0 / med_pipe, 3),
                    "speedup_pipeline": round(sp_pipe, 4),
                    "eficiencia_pipeline": round(ef_pipe, 4),
                    "speedup_lanzamiento": round(sp_lanz, 4),
                    "eficiencia_lanzamiento": round(ef_lanz, 4),
                    "bytes_transferidos": corridas[0]["bytes_transferidos"],
                    "mae": round(corridas[0]["mae"], 4) if corridas[0]["mae"] is not None else "",
                    "rmse": round(corridas[0]["rmse"], 4) if corridas[0]["rmse"] is not None else "",
                    "invalidos_pct": round(corridas[0]["invalidos_pct"], 2),
                    "salida_identica": todas_ok,
                }
                resumen.append(fila_res)
                print(
                    f"  {modo:<15} p={p} | pipe={med_pipe:.4f}s (S={sp_pipe:.2f}x, E={ef_pipe*100:.1f}%) "
                    f"| lanz={med_lanz:.4f}s (S={sp_lanz:.2f}x) | OK={todas_ok}"
                )

    # Guardar CSVs
    det_csv = RESULTS_DIR / "experimentos_detalle.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detalle[0].keys()))
        writer.writeheader()
        writer.writerows(detalle)

    res_csv = RESULTS_DIR / "experimentos_resumen.csv"
    with open(res_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        writer.writeheader()
        writer.writerows(resumen)

    print(f"\nTablas guardadas en:\n  - {det_csv}\n  - {res_csv}")


if __name__ == "__main__":
    main()