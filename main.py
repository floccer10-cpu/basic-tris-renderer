from model import Model
from PIL import Image
def main(w : int = 8000, h : int = 8000, s : float = 7.0e4, src : str = "model.obj", out : str = "img.png") -> int:
    model = Model(src)
    buf = model.rasterize(W = w, H = h, scale = s)
    img = Image.fromarray(buf)
    img.show()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
