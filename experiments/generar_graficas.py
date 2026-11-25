"""Genera las gráficas de rendimiento y mapas de disparidad/error del Proyecto 5."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "experiments" / "figures"
RESUMEN_CSV = RESULTS_DIR / "experimentos_resumen.csv"


def cargar_resumen():
    with open(RESUMEN_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def graficar_speedup_y_eficiencia(filas):
    escenarios = [
        ("small", "32", "Pequeña (225×187), D=32"),
        ("small", "64", "Pequeña (225×187), D=64"),
        ("large", "32", "Grande (450×375), D=32"),
        ("large", "64", "Grande (450×375), D=64"),
    ]

    # Figura 1: Speedup (Pipeline)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (ds, d, titulo) in zip(axes.flat, escenarios):
        sub = [r for r in filas if r["dataset"] == ds and r["max_disp"] == d]
        for modo, color, marcador in [("multiprocessing", "#1f77b4", "o"), ("mpi", "#d62728", "s")]:
            pts = sorted([r for r in sub if r["modo"] == modo], key=lambda x: int(x["procesos"]))
            if not pts:
                continue
            ps = [int(r["procesos"]) for r in pts]
            sp = [float(r["speedup_pipeline"]) for r in pts]
            ax.plot(ps, sp, marker=marcador, linewidth=2, color=color, label=modo.upper())
        ax.plot([1, 2, 4], [1, 2, 4], "--", color="gray", alpha=0.7, label="Ideal (S=p)")
        ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_xlabel("Procesos (p)")
        ax.set_ylabel("Speedup S(p)")
        ax.set_xticks([1, 2, 4])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Escalabilidad Fuerte: Speedup del Pipeline (Cómputo + Comunicación + Ensamblado)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_speedup_pipeline.png", dpi=300)
    plt.close()

    # Figura 2: Eficiencia (Pipeline)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (ds, d, titulo) in zip(axes.flat, escenarios):
        sub = [r for r in filas if r["dataset"] == ds and r["max_disp"] == d]
        for modo, color, marcador in [("multiprocessing", "#1f77b4", "o"), ("mpi", "#d62728", "s")]:
            pts = sorted([r for r in sub if r["modo"] == modo], key=lambda x: int(x["procesos"]))
            if not pts:
                continue
            ps = [int(r["procesos"]) for r in pts]
            ef = [float(r["eficiencia_pipeline"]) * 100.0 for r in pts]
            ax.plot(ps, ef, marker=marcador, linewidth=2, color=color, label=modo.upper())
        ax.axhline(100.0, color="gray", linestyle="--", alpha=0.7, label="Ideal (100%)")
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_xlabel("Procesos (p)")
        ax.set_ylabel("Eficiencia E(p) [%]")
        ax.set_xticks([1, 2, 4])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Eficiencia Paralela por Tamaño de Imagen y Rango de Disparidad", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_eficiencia_pipeline.png", dpi=300)
    plt.close()


def graficar_overhead_y_cruce(filas):
    # Figura 3: Tiempo de pipeline vs tiempo desde lanzamiento (p=4 vs secuencial)
    etiquetas = ["Pequeña\nD=32", "Pequeña\nD=64", "Grande\nD=32", "Grande\nD=64"]
    pares = [("small", "32"), ("small", "64"), ("large", "32"), ("large", "64")]

    seq_pipe, mp4_pipe, mpi4_pipe = [], [], []
    seq_lanz, mp4_lanz, mpi4_lanz = [], [], []

    for ds, d in pares:
        def get_val(modo, p, col):
            for r in filas:
                if r["dataset"] == ds and r["max_disp"] == d and r["modo"] == modo and int(r["procesos"]) == p:
                    return float(r[col])
            return 0.0

        seq_pipe.append(get_val("secuencial", 1, "t_pipeline_s"))
        mp4_pipe.append(get_val("multiprocessing", 4, "t_pipeline_s"))
        mpi4_pipe.append(get_val("mpi", 4, "t_pipeline_s"))

        seq_lanz.append(get_val("secuencial", 1, "t_lanzamiento_s"))
        mp4_lanz.append(get_val("multiprocessing", 4, "t_lanzamiento_s"))
        mpi4_lanz.append(get_val("mpi", 4, "t_lanzamiento_s"))

    x = np.arange(len(etiquetas))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.bar(x - w, seq_pipe, w, label="Secuencial (p=1)", color="#7f7f7f")
    ax1.bar(x, mp4_pipe, w, label="Multiprocessing (p=4)", color="#1f77b4")
    ax1.bar(x + w, mpi4_pipe, w, label="MPI (p=4)", color="#d62728")
    ax1.set_title("Tiempo interno del Pipeline (s)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas)
    ax1.set_ylabel("Tiempo (s) — menor es mejor")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    ax2.bar(x - w, seq_lanz, w, label="Secuencial (p=1)", color="#7f7f7f")
    ax2.bar(x, mp4_lanz, w, label="Multiprocessing (p=4)", color="#1f77b4")
    ax2.bar(x + w, mpi4_lanz, w, label="MPI (p=4)", color="#d62728")
    ax2.set_title("Tiempo total desde lanzamiento (incl. arranque SO/MPI)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas)
    ax2.set_ylabel("Tiempo (s) — menor es mejor")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    fig.suptitle("Punto de Cruce: Ganancia de Cómputo vs. Overhead de Coordinación y Arranque", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_punto_cruce_overhead.png", dpi=300)
    plt.close()


def graficar_mapas_disparidad():
    # Figura 4: Visualización cualitativa y mapa de error (dataset grande, D=64)
    ds_dir = ROOT / "data" / "middlebury_cones"
    disp_path = RESULTS_DIR / "mapas" / "large_d64_mpi_p4.png"
    if not disp_path.is_file():
        disp_path = RESULTS_DIR / "mapas" / "large_d64_secuencial_p1.png"

    left = np.array(Image.open(ds_dir / "left.png").convert("L"))
    gt = np.array(Image.open(ds_dir / "disp_gt.png").convert("L"), dtype=np.float32) / 4.0
    est = np.array(Image.open(disp_path).convert("L"), dtype=np.float32)
    err = np.abs(est - gt)
    err[:, :64] = 0.0  # Franja sin correspondencia

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].imshow(left, cmap="gray")
    axes[0].set_title("Imagen Izquierda")

    im1 = axes[1].imshow(gt, cmap="plasma", vmin=0, vmax=max(32, gt.max()))
    axes[1].set_title("Disparidad Referencia (GT)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(est, cmap="plasma", vmin=0, vmax=max(32, gt.max()))
    axes[2].set_title("Disparidad Estimada (MPI p=4)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(err, cmap="inferno", vmin=0, vmax=10)
    axes[3].set_title("Error Absoluto |Est - GT|")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    fig.suptitle("Evaluación Cualitativa de Correspondencia Estéreo (450×375, W=7, D=64)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_mapas_disparidad_error.png", dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filas = cargar_resumen()
    graficar_speedup_y_eficiencia(filas)
    graficar_overhead_y_cruce(filas)
    graficar_mapas_disparidad()
    print("Gráficas generadas en:", FIGURES_DIR)


if __name__ == "__main__":
    main()