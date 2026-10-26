"""Provided machinery for Lab 5, Aligning Two Models That Never Met.

The notebook downloads this file from the public notebooks repo (labs/data/lab5/, copied there by
make_lab5_data.py) and checks its digest. The clustering code is the pilot's (_workfiles/pilot/lab5/pilotlib5.py):
scikit-learn's k-means with k-means++ seeding, run to convergence, as in the authors' code, and the TA's best-swap
search over cluster matchings.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic CUDA matmuls

import hashlib  # noqa: E402
import io  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KMEANS_THREADS = min(4, os.cpu_count() or 1)
PHOTO_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"


# ---------------------------------------------------------------- data

def prep(x, mean):
    """Center on a training mean and scale every row to unit length."""
    x = x.to(DEVICE).float() - mean
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-10)


class Data:
    """The bundle, ready to use. Unpaired sets X and Y fit the bridges; the held-out pairs VX[i] and VY[i] belong
    to the same photo and only score them; CIFAR-10 is scored by its ten class sentences."""

    def __init__(self, folder):
        folder = pathlib.Path(folder)
        A = np.load(folder / "lab5_embeddings.npz", allow_pickle=False)
        D = {k: torch.from_numpy(A[k]) for k in A.files}
        self.raw = D
        self.mean_x = D["images"].to(DEVICE).float().mean(0, keepdim=True)
        self.mean_y = D["captions"].to(DEVICE).float().mean(0, keepdim=True)
        self.X, self.Y = prep(D["images"], self.mean_x), prep(D["captions"], self.mean_y)
        self.VX, self.VY = prep(D["val_images"], self.mean_x), prep(D["val_captions"], self.mean_y)
        self.PX, self.PY = prep(D["pair_images"], self.mean_x), prep(D["pair_captions"], self.mean_y)
        self.CX, self.CP = prep(D["cifar_images"], self.mean_x), prep(D["cifar_prompts"], self.mean_y)
        self.CL = D["cifar_labels"].to(DEVICE).long()
        M = np.load(folder / "lab5_maps.npz", allow_pickle=False)
        self.shipped = {k: torch.from_numpy(M[k]).to(DEVICE) for k in M.files}
        T = json.loads((folder / "lab5_text.json").read_text(encoding="utf-8"))
        self.ids, self.captions, self.kind = T["val_ids"], T["val_all_text"], T["val_kind"]
        self.classes = T["cifar_classes"]

    def kind_sets(self, kind):
        """6,000 photos of one kind and the captions of 6,000 other photos of that kind, centered on their own
        means, as the lab fits them."""
        D = self.raw
        mx = D[f"{kind}_images"].to(DEVICE).float().mean(0, keepdim=True)
        my = D[f"{kind}_captions"].to(DEVICE).float().mean(0, keepdim=True)
        return prep(D[f"{kind}_images"], mx), prep(D[f"{kind}_captions"], my), mx, my


def fetch(name, digest, base, folder):
    """Download one file into folder unless an intact copy is there; stop if its digest differs."""
    path = pathlib.Path(folder) / name
    if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest):
        path.parent.mkdir(parents=True, exist_ok=True)
        if base.startswith("http"):
            urllib.request.urlretrieve(base + name, path)
        else:
            path.write_bytes((pathlib.Path(base) / name).read_bytes())
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"{name} did not download intact. Delete {path} and run this cell again.")
    return path


# ---------------------------------------------------------------- scoring

def positions(W, VX, VY):
    """For each held-out photo, where its own caption lands among all the held-out captions (1 = first)."""
    sim = (VX @ W) @ VY.T
    return (sim > sim.diag()[:, None]).sum(1) + 1


def cifar_correct(W, data):
    """For each CIFAR-10 photo, 1 if the bridge labels it correctly from the ten class sentences, else 0."""
    return (((data.CX @ W) @ data.CP.T).argmax(1) == data.CL).float()


def cifar_accuracy(W, data):
    """Share of CIFAR-10 photos the bridge labels correctly from the ten class sentences."""
    return cifar_correct(W, data).mean().item()


_ANIMAL = None


def mentions_animal(text):
    """Whether a caption names an animal (the animals COCO labels, and a few everyday words for them)."""
    global _ANIMAL
    if _ANIMAL is None:
        import re
        _ANIMAL = re.compile(r"\b(animals?|dogs?|puppy|puppies|cats?|kittens?|kitty|birds?|horses?|cows?|cattle|"
                             r"sheep|zebras?|giraffes?|elephants?|bears?)\b", re.I)
    return bool(_ANIMAL.search(text))


def paired_interval(a, b, z=1.959964):
    """Mean of a - b over the same items, with its 95% interval."""
    d = (a - b).double()
    m, se = d.mean().item(), d.std(unbiased=True).item() / math.sqrt(len(d))
    return m, m - z * se, m + z * se


# ---------------------------------------------------------------- photos

_PHOTOS = {}


def photo(coco_id, size=220):
    """A thumbnail from COCO's image server, or None if it cannot be fetched."""
    if coco_id not in _PHOTOS:
        try:
            from PIL import Image
            with urllib.request.urlopen(PHOTO_URL.format(coco_id), timeout=20) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
            im.thumbnail((size, size))
            _PHOTOS[coco_id] = im
        except Exception:
            _PHOTOS[coco_id] = None
    return _PHOTOS[coco_id]


def show_photos(coco_ids, titles, columns=4):
    """Draw thumbnails in a grid; a photo that cannot be fetched leaves its title only."""
    import matplotlib.pyplot as plt
    rows = math.ceil(len(coco_ids) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, cid, title in zip(axes.flat, coco_ids, titles):
        im = photo(cid)
        if im is not None:
            ax.imshow(im)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def show_retrieval(k, i, j, data, position):
    """Retrieval k of the audit: held-out photo i beside photo j, whose captions the bridge put first."""
    show_photos([data.ids[i], data.ids[j]], [f"{k}. the photo", f"{k}. whose captions came first"], 2)
    print(f"{k}. Its own captions (position {position:,}):")
    for c in data.captions[i]:
        print("      " + c)
    if i == j:
        print("   The bridge put its own captions first.")
    else:
        print("   The captions the bridge put first:")
        for c in data.captions[j]:
            print("      " + c)


# ---------------------------------------------------------------- the three puzzles

_POINTS = torch.tensor([[0.0, 0.0], [2.0, 0.3], [3.2, 1.8], [1.1, 2.6], [-0.8, 1.5], [0.7, 1.0]], dtype=torch.float64)
_POINTS = _POINTS - _POINTS.mean(0)
_LETTERS = "ABCDEF"
_HIDDEN_TURN = 125.0                                   # degrees
_SHUFFLE = [3, 0, 5, 1, 4, 2]                          # copy label k + 1 sits on point _SHUFFLE[k]


def _turn(points, degrees):
    a = math.radians(degrees)
    R = torch.tensor([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]], dtype=torch.float64)
    return points @ R


def _agreement(P, Q, p):
    """How well the relations among the points of P agree with those among Q under the matching p (P[i] to Q[p[i]]).
    The same quantity as task 3's agreement, for points instead of cluster centers."""
    KP, KQ = P @ P.T, Q @ Q.T
    return (KP * KQ[p][:, p]).sum().item()


def _draw(ax, pts, labels, color, marker="o"):
    ax.scatter(pts[:, 0], pts[:, 1], s=140, c=color, marker=marker, edgecolors="black", zorder=3)
    for (x, y), lab in zip(pts.tolist(), labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(9, 7), fontsize=12, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.grid(alpha=0.3)


def puzzle_turn(angle):
    """Puzzle 1: the copy, turned back by `angle` degrees, over the original."""
    import matplotlib.pyplot as plt
    copy = _turn(_POINTS, _HIDDEN_TURN)
    shown = _turn(copy, -angle)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    _draw(ax, _POINTS, _LETTERS, "#9ecae1")
    _draw(ax, shown, [c.lower() for c in _LETTERS], "#fdae6b", marker="s")
    ax.set_title(f"original (circles) and the copy turned back {angle:g} degrees (squares)", fontsize=9)
    plt.show()
    d0, d1 = torch.cdist(_POINTS, _POINTS), torch.cdist(shown, shown)
    print(f"Distance from A to D: original {d0[0, 3]:.3f}, copy {d1[0, 3]:.3f}. "
          f"Largest change in any of the 15 distances: {(d0 - d1).abs().max():.2e}")
    gap = (shown - _POINTS).norm(dim=1).mean().item()
    print("The copy lines up with the original." if gap < 0.05 else f"Average gap between matching points: {gap:.2f}")


def puzzle_hidden():
    """Puzzle 2: the copy with its labels shuffled to numbers."""
    import matplotlib.pyplot as plt
    copy = _turn(_POINTS, _HIDDEN_TURN)
    numbered = [None] * 6
    for k, i in enumerate(_SHUFFLE):
        numbered[i] = str(k + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    _draw(axes[0], _POINTS, _LETTERS, "#9ecae1")
    axes[0].set_title("the original, labeled A to F", fontsize=10)
    _draw(axes[1], copy, numbered, "#fdae6b", marker="s")
    axes[1].set_title("the copy, turned and relabeled 1 to 6", fontsize=10)
    plt.show()


def _copy_in_number_order():
    copy = _turn(_POINTS, _HIDDEN_TURN)
    return copy[torch.tensor(_SHUFFLE)]                  # row k is the point labeled k + 1


def check_guess(guess):
    """Score a guessed matching, given as the numbers that A to F go to."""
    if sorted(guess) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("GUESS must use each of the numbers 1 to 6 once.")
    Q = _copy_in_number_order()
    p = torch.tensor([g - 1 for g in guess])
    truth = [_SHUFFLE.index(i) + 1 for i in range(6)]
    best = _agreement(_POINTS, Q, torch.tensor([t - 1 for t in truth]))
    print(f"Agreement of your matching: {_agreement(_POINTS, Q, p):.2f} (the best possible is {best:.2f})")
    print("That is the hidden matching." if list(guess) == truth else
          f"{sum(g == t for g, t in zip(guess, truth))} of 6 letters are matched correctly.")


def search_matchings():
    """Try all 720 matchings of the copy and rank them by agreement."""
    Q = _copy_in_number_order()
    scored = sorted(((_agreement(_POINTS, Q, torch.tensor(p)), [i + 1 for i in p])
                     for p in itertools.permutations(range(6))), reverse=True)
    truth = [_SHUFFLE.index(i) + 1 for i in range(6)]
    for rank, (a, p) in enumerate(scored[:3], 1):
        tag = "  <- the hidden matching" if p == truth else ""
        print(f"{rank}. A to F go to {p}: agreement {a:.2f}{tag}")
    print(f"... and 717 more, down to {scored[-1][0]:.2f}")


def puzzle_square():
    """Puzzle 3: four corners of a square, matched to a turned copy of themselves."""
    import matplotlib.pyplot as plt
    P = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)
    Q = _turn(P, 30.0)
    scored = sorted(((round(_agreement(P, Q, torch.tensor(p)), 6), [i + 1 for i in p])
                     for p in itertools.permutations(range(4))), reverse=True)
    top = [p for a, p in scored if a == scored[0][0]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    _draw(axes[0], P * 1.5, "ABCD", "#9ecae1")
    _draw(axes[1], Q * 1.5, "1234", "#fdae6b", marker="s")
    plt.show()
    print(f"{len(top)} of the 24 matchings tie for the best agreement, {scored[0][0]:.2f}:")
    for p in top:
        print("   A to D go to", p)


# ---------------------------------------------------------------- the clustering machinery

def centered_kernel(Z):
    """The table of inner products of the rows of Z, with row and column means removed."""
    K = Z @ Z.T
    return K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()


def kmeans(Z, C, gen):
    """scikit-learn's k-means: k-means++ seeding, run to convergence, as in the authors' code. A seed drawn from
    gen fixes the result, and fixed threads keep it the same from run to run."""
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=gen.device))
    with threadpool_limits(KMEANS_THREADS):
        model = KMeans(n_clusters=C, init="k-means++", random_state=seed).fit(Z.double().cpu().numpy())
    return torch.as_tensor(model.cluster_centers_, dtype=Z.dtype, device=Z.device)


def match_clusters(A, B, gen, restarts=2000, max_swaps=200):
    """The pairing p that maximizes the agreement of A and B, by best-swap search from many random starts."""
    KA, KB = centered_kernel(A).double(), centered_kernel(B).double()
    C, dA, rows = len(KA), KA.diag(), torch.arange(restarts, device=A.device)
    p = torch.stack([torch.randperm(C, generator=gen, device=A.device) for _ in range(restarts)])
    for _ in range(max_swaps):
        Bp = KB[p[:, :, None], p[:, None, :]]
        G = KA @ Bp.transpose(1, 2)
        dG, dB = G.diagonal(dim1=1, dim2=2), Bp.diagonal(dim1=1, dim2=2)
        gain = 2 * (G + G.transpose(1, 2) - dG[:, :, None] - dG[:, None, :]
                    - (dA[None, :, None] - KA.T[None]) * (Bp.transpose(1, 2) - dB[:, :, None])
                    - (KA[None] - dA[None, None, :]) * (dB[:, None, :] - Bp))
        gain = gain + (dA[None, :, None] - dA[None, None, :]) * (dB[:, None, :] - dB[:, :, None])
        best, where = gain.flatten(1).max(1)              # the best single swap for every start
        go = best > 1e-12
        if not go.any():
            break
        r, s = where // C, where % C
        pr, ps = p[rows, r].clone(), p[rows, s].clone()
        p[rows[go], r[go]], p[rows[go], s[go]] = ps[go], pr[go]
    value = (KA[None] * KB[p[:, :, None], p[:, None, :]]).sum((1, 2))
    return p[value.argmax()]


def hungarian(scores):
    """One-to-one matching with the highest total score. Returns (rows, columns)."""
    rows, columns = linear_sum_assignment(-scores.double().cpu().numpy())
    return torch.as_tensor(rows, device=scores.device), torch.as_tensor(columns, device=scores.device)


def matching_score(Xs, Ys, W, batch=4000):
    """Mean similarity of one-to-one matched pairs under W, on unpaired data only."""
    xb, yb = Xs[:batch], Ys[:batch]
    rows, columns = hungarian((xb @ W) @ yb.T)
    return ((xb[rows] @ W) * yb[columns]).sum(1).mean().item()


# ---------------------------------------------------------------- saving fits

def stamp(*functions, **settings):
    """A short fingerprint of the code and settings a fit depends on. A saved fit is reused only while it matches, so
    changing a function or a setting changes the experiment you see."""
    import inspect
    parts = []
    for f in functions:
        try:
            parts.append(inspect.getsource(f))
        except (OSError, TypeError):                       # no source to read: the compiled code, stable across reruns
            parts.append(f.__code__.co_code.hex() + repr([c for c in f.__code__.co_consts if not hasattr(c, "co_code")]))
    parts.append(json.dumps(settings, sort_keys=True, default=str))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def device_name():
    return torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "a CPU"


class Saved:
    """Fits saved as they finish, in Google Drive if it is mounted. Run all reuses a saved fit only while the code,
    data and settings it was fitted with are unchanged."""

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def get(self, name, stamp):
        path = self.folder / f"{name}.pt"
        if not path.exists():
            return None
        out = torch.load(path, weights_only=True, map_location=DEVICE)
        if out.get("stamp") != stamp:
            print(f"{name}: your code or settings changed since this fit was saved, so it runs again")
            return None
        print(f"{name}: loaded the fit saved in {path}, made on {out.get('device', 'an unrecorded device')}")
        return out

    def put(self, name, value, stamp):
        value = {**value, "stamp": stamp, "device": device_name()}
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in value.items()}, self.folder / f"{name}.pt")
        return value
