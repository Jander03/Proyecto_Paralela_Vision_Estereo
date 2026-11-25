"""Núcleo de correspondencia estéreo por ventanas (SAD), manejo de halos y métricas."""
from pathlib import Path
import numpy as np
from PIL import Image


def cargar_par_estereo(carpeta: Path):
    """Carga imágenes izquierda, derecha (grises) y ground truth si existe."""
    left = np.array(Image.open(carpeta / "left.png").convert("L"), dtype=np.int32)
    right = np.array(Image.open(carpeta / "right.png").convert("L"), dtype=np.int32)
    gt_path = carpeta / "disp_gt.png"
    gt = None
    if gt_path.is_file():
        # Middlebury 2003 almacena disparidad escalada x4
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 4.0
    return left, right, gt


def calcular_franja_sad(left_sub: np.ndarray, right_sub: np.ndarray,
                        y0: int, y1: int, in_y0: int, in_y1: int,
                        max_disp: int, win_size: int) -> np.ndarray:
    """Calcula el mapa de disparidad SAD para la franja [y0, y1) usando halos."""
    r = win_size // 2
    h_out = y1 - y0
    w = left_sub.shape[1]

    # Relleno exacto en bordes superior/inferior para igualar la ejecución secuencial
    pad_top = r - (y0 - in_y0)
    pad_bot = r - (in_y1 - y1)
    L_pad = np.pad(left_sub, ((pad_top, pad_bot), (r, r)), mode="edge")
    R_pad = np.pad(right_sub, ((pad_top, pad_bot), (r, r)), mode="edge")

    best_cost = np.full((h_out, w), np.iinfo(np.int32).max, dtype=np.int32)
    best_disp = np.zeros((h_out, w), dtype=np.uint8)

    for d in range(max_disp):
        # Desplazamiento horizontal de la imagen derecha
        if d == 0:
            diff = np.abs(L_pad - R_pad)
        else:
            R_shift = np.empty_like(R_pad)
            R_shift[:, :d] = R_pad[:, :1]
            R_shift[:, d:] = R_pad[:, :-d]
            diff = np.abs(L_pad - R_shift)

        # Suma de ventana W x W
        cost = np.zeros((h_out, w), dtype=np.int32)
        for dy in range(win_size):
            for dx in range(win_size):
                cost += diff[dy:dy + h_out, dx:dx + w]

        mask = cost < best_cost
        best_cost[mask] = cost[mask]
        best_disp[mask] = d

    return best_disp


def dividir_franjas_con_halo(left: np.ndarray, right: np.ndarray,
                             num_partes: int, win_size: int):
    """Divide las imágenes en franjas horizontales incluyendo halos de tamaño r."""
    h = left.shape[0]
    r = win_size // 2
    cortes = np.linspace(0, h, num_partes + 1, dtype=int)
    franjas = []
    for i in range(num_partes):
        y0, y1 = int(cortes[i]), int(cortes[i + 1])
        in_y0 = max(0, y0 - r)
        in_y1 = min(h, y1 + r)
        franjas.append((
            left[in_y0:in_y1, :].copy(),
            right[in_y0:in_y1, :].copy(),
            y0, y1, in_y0, in_y1
        ))
    return franjas


def evaluar_disparidad(disp: np.ndarray, gt: np.ndarray, max_disp: int):
    """Calcula MAE, RMSE y porcentaje de píxeles inválidos (oclusión/borde)."""
    validos = (gt > 0) if gt is not None else np.ones_like(disp, dtype=bool)
    # Ignorar franja izquierda sin correspondencia posible (x < max_disp)
    validos[:, :max_disp] = False
    invalidos_pct = float(100.0 * (1.0 - np.mean(validos)))

    if gt is None or not np.any(validos):
        return {"mae": None, "rmse": None, "invalidos_pct": invalidos_pct}

    error = np.abs(disp[validos].astype(np.float32) - gt[validos])
    return {
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "invalidos_pct": invalidos_pct,
    }