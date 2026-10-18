from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image
from math import cos

@njit(fastmath = True)
def bar_coord(x0, y0, z0, x1, y1, z1, x2, y2, z2, x, y):

    div = (x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / div
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / div
    lambda2 =  1.0 - lambda0 - lambda1
    
    return (lambda0, lambda1, lambda2)

@njit(fastmath = True)
def triangle(buf, z_buf, W, H, light_cosine, x0, y0, z0,
             x1, y1, z1, x2, y2, z2):

    x_min = int(max(0, min(x0, x1, x2)))
    y_min = int(max(0, min(y0, y1, y2)))
    x_max = int(min(W, max(x0, x1, x2)))
    y_max = int(min(H, max(y0, y1, y2)))

    rgb = np.random.randint(low = 128, high = 256, size = (1, 3))*-light_cosine
    for _x in range(x_min, min(x_max + 1, W)):
        for _y in range(y_min, min(y_max + 1, H)):
            x = _x + 0.5; y = _y + 0.5
            lambda0, lambda1, lambda2 = bar_coord(x0, y0, z0, x1, y1, z1, x2, y2, z2, x, y)

            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z < z_buf[_y, _x]:
                    buf[_y, _x] = rgb
                    z_buf[_y, _x] = z
class Model:
    vertices : np.ndarray
    edges    : np.ndarray
    faces    : np.ndarray
    def __init__(self, path):
        # validating path
        p = self._validate_path(path)

        # parsing file
        self._parse_file(p)
            
        # faces to edges
        self._faces_to_edges()

    def rasterize(self, W : int, H : int, scale : float,
                  cam_dir : list[int] = None, light_dir : list[int] = None) -> np.ndarray:
        if cam_dir is None:
            cam_dir = [0, 0, -1]
        if light_dir is None:
            light_dir = [1/3, 1/2, 1]
                
        cam_dir /= np.linalg.norm(cam_dir)
        light_dir /= np.linalg.norm(light_dir)
        buf = np.zeros((H, W, 3), dtype=np.uint8) # экран
        z_buf = np.full((H, W), np.inf)
        screen = self.vertices[:, :]*scale + [W/2, 0, 0]
        
        for face in self.faces:
            first, second, third = face
            x0, y0, z0 = screen[first]
            x1, y1, z1 = screen[second]
            x2, y2, z2 = screen[third]

            normal = np.cross( (x1 - x2, y1 - y2, z1 - z2), (x1 - x0, y1 - y0, z1 - z0) )
            cosine = np.dot(normal, light_dir)/np.linalg.norm(normal)
            if cosine > 0:
                continue
            
            triangle(buf, z_buf, W, H, cosine, x0, y0, z0, x1, y1, z1, x2, y2, z2)
        buf = np.flip(buf, axis = (0, 1))
        return buf

    def _faces_to_edges(self) -> None:
        self.edges = np.concatenate([
        self.faces[:, [0, 1]],
        self.faces[:, [1, 2]],
        self.faces[:, [0, 2]]
        ], axis = 0)
        self.edges = np.sort(self.edges, axis = 1)
        self.edges = np.unique(self.edges, axis = 0)

    @staticmethod
    def _validate_path(path : str | Path) -> Path:
        if not str(path).strip():
            raise ValueError("path can't be empty")
        p = Path(path)
        if not p.exists():
            raise ValueError("file doesn't exist")
        if not p.is_file():
            raise IsADirectoryError("not a file")
        suf = p.suffix
        if suf.lower() != ".obj":
            raise ValueError(f"expected .obj file, got {suf}")
        return p

    def _parse_file(self, p: Path) -> None:
        with p.open(mode='r', errors="replace", encoding="utf-8") as f:
            V: list[tuple[float, float, float]]  = []
            F: list[tuple[int, int, int]] = []
            for lineno, raw in enumerate(f, start = 1):
                parts = raw.split()
                if parts[0] == 'v':
                    if len(parts) < 4:
                        raise ValueError(f"{p}:{lineno}: vertex must have 3 coordinates")
                    try:
                        x, y, z = float(parts[1]),  float(parts[2]), float(parts[3])
                        V.append((x, y, z))
                    except ValueError as e:
                        raise ValueError(f"{p}:{lineno}: bad vertex: {e}") from e
           
                elif parts[0] == 'f':
                    idx: list[int] = []
                    for token in parts[1:]:
                        head = token.split("/", 1)[0] # первая координата - индекс вершины модели
                        if not head:
                            raise ValueError(f"{p}:{lineno}: empty face index")
                        try:
                            idx.append(int(head) - 1)
                        except ValueError as e:
                            raise ValueError(f"{p}:{lineno}: bad face index: {e}") from e
                    if len(idx) < 3:
                        raise ValueError(f"{p}:{lineno}: face must point to at least 3 indices")
                    if any(i < 0 for i in idx):
                        raise ValueError(f"{p}:{lineno}: face indices must be natural numbers")
                    for i in range(1, len(idx) - 1):
                        F.append((idx[0], idx[i], idx[i+1]))
            if not V:
                raise ValueError(f"{p}: no vertices found")
            if not F:
                raise ValueError(f"{p}: no faces found")

            self.vertices = np.asarray(V, dtype = np.float64)
            self.faces = np.asarray(F, dtype = np.uint32)
            if self.faces.max() >= self.vertices.shape[0]:
                raise ValueError(f"{p}: faces reference vertex {self.faces.max() + 1}, only {self.vertices.shape[0]} vertices defined")

