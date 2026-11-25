from pathlib import Path
import urllib.request
import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
MIDDLEBURY_DIR = DATA_DIR / "middlebury_cones"

URLS = {
    "left.png": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png",
    "right.png": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png",
    "disp_gt.png": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/disp2.png",
}

def generar_respaldo_sintetico(destino: Path):
    print("Generando par estéreo sintético de respaldo...")
    rng = np.random.default_rng(42)
    h, w = 375, 450
    fondo = rng.integers(40, 215, size=(h, w), dtype=np.uint8)
    left = fondo.copy()
    right = fondo.copy()
    gt = np.full((h, w), 8, dtype=np.uint8)
    right[:, :-8] = left[:, 8:]

    y1, y2, x1, x2 = 80, 220, 120, 260
    tex1 = rng.integers(0, 255, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    left[y1:y2, x1:x2] = tex1
    right[y1:y2, x1 - 24:x2 - 24] = tex1
    gt[y1:y2, x1:x2] = 24

    y1, y2, x1, x2 = 180, 320, 260, 390
    tex2 = rng.integers(0, 255, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    left[y1:y2, x1:x2] = tex2
    right[y1:y2, x1 - 16:x2 - 16] = tex2
    gt[y1:y2, x1:x2] = 16

    Image.fromarray(left).save(destino / "left.png")
    Image.fromarray(right).save(destino / "right.png")
    Image.fromarray(gt * 4).save(destino / "disp_gt.png")

def descargar():
    MIDDLEBURY_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for nombre, url in URLS.items():
        ruta = MIDDLEBURY_DIR / nombre
        if ruta.is_file():
            print(f"[OK] Ya existe: {ruta.name}")
            continue
        try:
            print(f"Descargando {nombre} desde Middlebury...")
            urllib.request.urlretrieve(url, ruta)
        except Exception as e:
            print(f"[AVISO] No se pudo descargar {nombre}: {e}")
            ok = False
            break

    if not ok:
        generar_respaldo_sintetico(MIDDLEBURY_DIR)

    small_dir = DATA_DIR / "middlebury_cones_small"
    small_dir.mkdir(parents=True, exist_ok=True)
    for nombre in URLS.keys():
        with Image.open(MIDDLEBURY_DIR / nombre) as img:
            w, h = img.size
            resample = Image.NEAREST if "disp" in nombre else Image.BILINEAR
            img_small = img.resize((w // 2, h // 2), resample=resample)
            if "disp" in nombre:
                arr = (np.array(img_small, dtype=np.float32) / 2.0).astype(np.uint8)
                Image.fromarray(arr).save(small_dir / nombre)
            else:
                img_small.save(small_dir / nombre)

    print("Datos listos en:", MIDDLEBURY_DIR, "y", small_dir)

if __name__ == "__main__":
    descargar()
