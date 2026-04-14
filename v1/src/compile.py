"""Compile visual assets from input images and source code.

Run once before a live set. Pre-processes heavy operations (blur,
displacement, solarize, pixel sort) so the live renderer only needs
fast blending at runtime.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.transforms import (
    make_abstract, pixel_sort,
    radial_displace, fluid_displace, multi_pass_abstract, make_vignette,
)

RESOLUTION = (1920, 1080)
OUTPUT_DIR = "compiled"
IMAGE_GLOB = "input/images/*"
SOURCE_GLOBS = [
    "*.js", "*.ts", "*.jsx", "*.tsx",
    "*.py", "*.go", "*.rs", "*.rb",
    "*.java", "*.kt", "*.scala",
    "*.c", "*.cpp", "*.h", "*.hpp",
    "*.cs", "*.swift", "*.m",
    "*.sh", "*.bash", "*.zsh",
    "*.lua", "*.zig", "*.nim",
    "*.html", "*.css", "*.scss",
    "*.sql", "*.graphql",
    "*.ex", "*.exs", "*.erl",
    "*.hs", "*.ml", "*.clj",
    "*.r", "*.R", "*.jl",
]

W, H = RESOLUTION

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
_SOURCE_EXTS = {
    ".js", ".ts", ".jsx", ".tsx", ".py", ".go", ".rs", ".rb",
    ".java", ".kt", ".scala", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".swift", ".m", ".sh", ".bash", ".zsh",
    ".lua", ".zig", ".nim", ".html", ".css", ".scss",
    ".sql", ".graphql", ".ex", ".exs", ".erl",
    ".hs", ".ml", ".clj", ".r", ".R", ".jl",
}


# ==================== IMAGE LOADING ====================
def load_images(pattern=None):
    pattern = pattern or IMAGE_GLOB
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if os.path.splitext(p)[1].lower() in _IMAGE_EXTS]
    if not paths:
        print(f"ERROR: No images found matching '{pattern}'")
        return []
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
        images.append(np.array(img, dtype=np.uint8))
        print(f"  {p} -> {W}x{H}")
    return images


# ==================== SOURCE FILE COLLECTION ====================
def collect_source_files(paths=None):
    if paths:
        files = []
        for p in paths:
            p = os.path.expanduser(p)
            if os.path.isfile(p):
                files.append(p)
            elif os.path.isdir(p):
                for root, dirs, fnames in os.walk(p):
                    dirs[:] = [d for d in dirs if not d.startswith(".")
                               and d not in ("node_modules", "__pycache__", "venv", ".venv")]
                    for fn in sorted(fnames):
                        _, ext = os.path.splitext(fn)
                        if ext.lower() in _SOURCE_EXTS:
                            files.append(os.path.join(root, fn))
            else:
                matches = sorted(glob.glob(p, recursive=True))
                if matches:
                    files.extend(matches)
                else:
                    print(f"  WARNING: path not found: {p}")
                    sys.exit(1)
    else:
        files = []
        for pattern in SOURCE_GLOBS:
            files.extend(sorted(glob.glob(pattern)))

    seen = set()
    unique = []
    for f in files:
        abspath = os.path.abspath(f)
        if abspath not in seen:
            seen.add(abspath)
            unique.append(f)
    return unique


# ==================== CODE ANALYSIS ====================
def analyze_code(source_files):
    texts = []
    for f in source_files:
        try:
            with open(f) as fh:
                texts.append(fh.read())
            print(f"  {f} ({os.path.getsize(f)} bytes)")
        except (UnicodeDecodeError, OSError):
            pass
    combined = "\n".join(texts)
    if not source_files or not combined:
        print("  No source files found - using default features")
        return {"entropy": 0.5, "bracket_density": 0.3, "nesting_depth": 0.3,
                "line_variance": 0.5, "token_density": 0.3}

    freq = Counter(combined)
    length = len(combined)
    raw_ent = -sum((c / length) * math.log2(c / length) for c in freq.values())
    entropy = float(np.clip((raw_ent - 3.5) / 2.0, 0, 1))

    brackets = sum(combined.count(c) for c in "{}[]()<>")
    bracket_density = min(brackets / max(len(combined), 1) * 10, 1.0)

    depth, max_depth = 0, 0
    for ch in combined:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    nesting_depth = min(max_depth / 10.0, 1.0)

    lines = combined.split("\n")
    lengths = [len(ln) for ln in lines if ln.strip()]
    line_var = float(np.std(lengths) / max(np.mean(lengths), 1)) if lengths else 0.0

    tokens = re.findall(r"[a-zA-Z_]\w*", combined)
    tok_den = len(set(tokens)) / len(tokens) if tokens else 0.0

    return {
        "entropy": entropy,
        "bracket_density": bracket_density,
        "nesting_depth": nesting_depth,
        "line_variance": min(line_var, 1.0),
        "token_density": tok_den,
    }


# ==================== LANGUAGE DETECTION & PALETTES ====================
_PALETTE_JS = [
    (255, 136, 68), (255, 204, 68), (68, 221, 255),
    (255, 255, 255), (255, 170, 100), (200, 200, 210),
]
_PALETTE_PYTHON = [
    (68, 255, 136), (68, 136, 255), (255, 170, 68),
    (100, 255, 180), (130, 200, 255), (200, 210, 200),
]
_PALETTE_SYSTEMS = [
    (255, 68, 68), (170, 170, 170), (255, 136, 68),
    (200, 200, 200), (255, 100, 100), (190, 190, 190),
]
_PALETTE_WEB = [
    (255, 68, 170), (170, 68, 255), (68, 255, 204),
    (200, 130, 255), (255, 130, 200), (210, 200, 220),
]
_PALETTE_DEFAULT = [
    (255, 100, 100), (100, 255, 100), (100, 200, 255),
    (255, 200, 80), (200, 150, 255), (220, 220, 220),
]

_LANG_FAMILIES = {
    "js_ts":   {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"},
    "python":  {".py", ".pyw"},
    "systems": {".rs", ".c", ".cpp", ".h", ".hpp", ".go", ".zig"},
    "web":     {".html", ".css", ".scss", ".svg"},
    "jvm":     {".java", ".kt", ".scala"},
    "shell":   {".sh", ".bash", ".zsh"},
    "func":    {".hs", ".ml", ".clj", ".ex", ".exs", ".erl"},
}

_FAMILY_PALETTES = {
    "js_ts": _PALETTE_JS, "python": _PALETTE_PYTHON,
    "systems": _PALETTE_SYSTEMS, "web": _PALETTE_WEB,
    "jvm": _PALETTE_JS, "shell": _PALETTE_SYSTEMS,
    "func": _PALETTE_WEB,
}

_KEYWORDS_BASE = {
    "if", "else", "for", "while", "return", "break", "continue",
    "try", "catch", "throw", "finally", "switch", "case", "default",
    "true", "false", "null", "void",
}
_KEYWORDS_JS = _KEYWORDS_BASE | {
    "function", "const", "let", "var", "export", "import", "from",
    "class", "extends", "new", "this", "typeof", "instanceof",
    "async", "await", "yield", "of", "in", "delete",
    "interface", "type", "enum", "implements", "readonly",
    "undefined", "NaN", "Infinity", "Promise", "Array", "Object",
}
_KEYWORDS_PYTHON = _KEYWORDS_BASE | {
    "def", "class", "import", "from", "return", "elif", "pass",
    "with", "as", "in", "not", "and", "or", "is", "lambda",
    "yield", "async", "await", "raise", "except", "assert",
    "True", "False", "None", "self", "nonlocal", "global",
    "print", "range", "len", "dict", "list", "set", "tuple",
}
_KEYWORDS_SYSTEMS = _KEYWORDS_BASE | {
    "fn", "let", "mut", "pub", "struct", "enum", "impl", "trait",
    "use", "mod", "crate", "unsafe", "match", "loop", "ref",
    "static", "const", "extern", "sizeof", "typedef", "include",
    "int", "float", "double", "char", "long", "unsigned",
    "auto", "register", "volatile", "inline", "template", "namespace",
    "func", "go", "chan", "select", "defer", "package",
}
_KEYWORDS_WEB = _KEYWORDS_BASE | {
    "div", "span", "class", "id", "style", "href", "src", "alt",
    "html", "head", "body", "script", "link", "meta", "title",
    "display", "flex", "grid", "margin", "padding", "border",
    "color", "background", "font", "width", "height", "position",
    "hover", "active", "focus", "media", "keyframes", "animation",
}

_FAMILY_KEYWORDS = {
    "js_ts": _KEYWORDS_JS, "python": _KEYWORDS_PYTHON,
    "systems": _KEYWORDS_SYSTEMS, "web": _KEYWORDS_WEB,
    "jvm": _KEYWORDS_JS, "shell": _KEYWORDS_BASE,
    "func": _KEYWORDS_BASE,
}


def _detect_lang_family(filename):
    _, ext = os.path.splitext(filename)
    for family, exts in _LANG_FAMILIES.items():
        if ext.lower() in exts:
            return family
    return "default"


def _get_palette(family):
    return _FAMILY_PALETTES.get(family, _PALETTE_DEFAULT)


def _get_keywords(family):
    return _FAMILY_KEYWORDS.get(family, _KEYWORDS_BASE)


# ==================== CODE -> IMAGE RENDERING ====================
_TOKEN_RE = re.compile(
    r'[a-zA-Z_]\w*|"[^"]*"|\'[^\']*\'|#.*|//.*|[{}()\[\]<>]|'
    r'[=+\-*/%&|^~!<>]+|\d+\.?\d*|\s+|.'
)

_font_cache = {}


def _get_mono_font(size):
    if size in _font_cache:
        return _font_cache[size]
    mono_names = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in mono_names:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _token_color(token, palette=None, keywords=None):
    if palette is None:
        palette = _PALETTE_DEFAULT
    if keywords is None:
        keywords = _KEYWORDS_BASE
    stripped = token.strip()
    if not stripped:
        return palette[5]
    if stripped in keywords:
        return palette[0]
    if stripped.startswith(("#", "//", "/*", "'''", '"""')):
        return palette[1]
    if stripped.startswith(('"', "'", "`")):
        return palette[1]
    if any(c in stripped for c in "{}[]()<>"):
        return palette[3]
    if any(c in stripped for c in "=+-*/%&|^~!<>"):
        return palette[4]
    try:
        float(stripped)
        return palette[2]
    except ValueError:
        pass
    return palette[5]


def _render_code_terminal(text, name, palette, keywords):
    font = _get_mono_font(14)
    img = Image.new("RGB", (W, H), (10, 10, 15))
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox("M")
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1] + 4
    lines = [f"// === {name} ==="] + text.split("\n")
    y = 4
    for line in lines:
        if y + char_h > H:
            break
        tokens = _TOKEN_RE.findall(line)
        x = 8
        for tok in tokens:
            if x > W - 8:
                break
            color = _token_color(tok, palette, keywords)
            draw.text((x, y), tok, fill=color, font=font)
            tok_w = font.getlength(tok) if hasattr(font, "getlength") else len(tok) * char_w
            x += int(tok_w)
        y += char_h
    return np.array(img, dtype=np.uint8)


def _render_code_dense(text, name, palette, keywords):
    font = _get_mono_font(8)
    img = Image.new("RGB", (W, H), (5, 5, 10))
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox("M")
    char_w = max(bbox[2] - bbox[0], 1)
    char_h = max(bbox[3] - bbox[1] + 1, 1)
    cols = W // char_w
    rows = H // char_h
    lines = text.split("\n")
    while len(lines) < rows:
        lines = lines + lines
    y = 0
    for i, line in enumerate(lines[:rows]):
        if y + char_h > H:
            break
        color_idx = (i * 7) % len(palette)
        color = palette[color_idx]
        color = (color[0] // 2 + 40, color[1] // 2 + 40, color[2] // 2 + 40)
        draw.text((0, y), line[:cols], fill=color, font=font)
        y += char_h
    return np.array(img, dtype=np.uint8)



def _render_code_scatter(text, name, palette, keywords):
    img = Image.new("RGB", (W, H), (8, 8, 12))
    draw = ImageDraw.Draw(img)
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return np.array(img, dtype=np.uint8)
    if len(lines) < 3:
        lines = lines * 10
    rng = np.random.RandomState(hash(name) % (2**31))
    n_chunks = rng.randint(8, 13)
    for _ in range(n_chunks):
        chunk_len = rng.randint(2, max(3, min(6, len(lines))))
        start = rng.randint(0, max(1, len(lines) - chunk_len))
        chunk = lines[start:start + chunk_len]
        font_size = rng.randint(10, 37)
        font = _get_mono_font(font_size)
        x = rng.randint(-100, W - 100)
        y = rng.randint(-50, H - 50)
        bbox = font.getbbox("M")
        line_h = max(bbox[3] - bbox[1] + 2, 1)
        dim = min(1.0, font_size / 28.0)
        for j, line in enumerate(chunk):
            tokens = _TOKEN_RE.findall(line)
            tx = x
            for tok in tokens:
                if tx > W + 50:
                    break
                color = _token_color(tok, palette, keywords)
                color = tuple(int(c * dim) for c in color)
                draw.text((tx, y + j * line_h), tok, fill=color, font=font)
                tok_w = font.getlength(tok) if hasattr(font, "getlength") else len(tok) * 8
                tx += int(tok_w)
    return np.array(img, dtype=np.uint8)


def _render_code_blueprint(text, name, palette, keywords):
    """Cyan/white code on dark blue background with subtle grid lines."""
    font = _get_mono_font(13)
    img = Image.new("RGB", (W, H), (8, 12, 35))
    draw = ImageDraw.Draw(img)
    # Grid lines
    for gx in range(0, W, 40):
        draw.line([(gx, 0), (gx, H)], fill=(15, 25, 55), width=1)
    for gy in range(0, H, 40):
        draw.line([(0, gy), (W, gy)], fill=(15, 25, 55), width=1)
    bbox = font.getbbox("M")
    char_h = bbox[3] - bbox[1] + 3
    lines = text.split("\n")
    y = 6
    for i, line in enumerate(lines):
        if y + char_h > H:
            break
        tokens = _TOKEN_RE.findall(line)
        x = 12
        for tok in tokens:
            if x > W - 12:
                break
            stripped = tok.strip()
            if stripped in (keywords or _KEYWORDS_BASE):
                color = (180, 240, 255)
            elif any(c in stripped for c in "{}[]()<>"):
                color = (100, 180, 220)
            elif stripped.startswith(('"', "'", '`', '#', '//')):
                color = (60, 120, 160)
            else:
                color = (140, 210, 240)
            draw.text((x, y), tok, fill=color, font=font)
            tok_w = font.getlength(tok) if hasattr(font, "getlength") else len(tok) * 8
            x += int(tok_w)
        y += char_h
    return np.array(img, dtype=np.uint8)


def _render_code_neon(text, name, palette, keywords):
    """Large bright neon-colored code fragments on black."""
    img = Image.new("RGB", (W, H), (2, 2, 5))
    draw = ImageDraw.Draw(img)
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return np.array(img, dtype=np.uint8)
    rng = np.random.RandomState(hash(name) % (2**31))
    neon_colors = [
        (255, 50, 100), (50, 255, 150), (100, 150, 255),
        (255, 200, 50), (200, 100, 255), (50, 255, 255),
    ]
    n_blocks = rng.randint(5, 9)
    for _ in range(n_blocks):
        font_size = rng.randint(18, 48)
        font = _get_mono_font(font_size)
        bbox = font.getbbox("M")
        line_h = max(bbox[3] - bbox[1] + 2, 1)
        n_lines = rng.randint(1, max(2, min(5, len(lines))))
        start = rng.randint(0, max(1, len(lines) - n_lines))
        x = rng.randint(-80, W - 200)
        y = rng.randint(-20, H - 40)
        color = neon_colors[rng.randint(0, len(neon_colors))]
        for j in range(n_lines):
            line = lines[(start + j) % len(lines)]
            draw.text((x, y + j * line_h), line[:60], fill=color, font=font)
    return np.array(img, dtype=np.uint8)


_STYLE_FUNCS = {
    "term": _render_code_terminal,
    "dense": _render_code_dense,

    "scatter": _render_code_scatter,
    "blueprint": _render_code_blueprint,
    "neon": _render_code_neon,
}
_STYLES = ["term", "dense", "scatter", "blueprint", "neon"]
_MAX_CODE_IMAGES = 18


# ==================== MAIN ====================
def _save(key, data, keys_list, output_dir):
    np.savez_compressed(os.path.join(output_dir, f"{key}.npz"), data=data)
    keys_list.append(key)


def _compile_image(args):
    """Compile all variants for a single image. Runs in a worker process."""
    prefix, img, output_dir = args
    abstract = make_abstract(img)
    results = [(f"{prefix}_sharp", img), (f"{prefix}_abstract", abstract)]
    results.append((f"{prefix}_sorted", pixel_sort(abstract)))
    results.append((f"{prefix}_deep", multi_pass_abstract(abstract)))
    for amp, freq, ph in [(30, 3, 0), (60, 5, 1.5), (90, 8, 3.0)]:
        results.append((f"{prefix}_radial_{amp}", radial_displace(abstract, amp, freq, ph)))
    for amp, turb, ph in [(40, 2, 0), (80, 3, 2.5), (120, 4, 5.0)]:
        results.append((f"{prefix}_fluid_{amp}", fluid_displace(abstract, amp, turb, ph)))
    keys = []
    for key, data in results:
        np.savez_compressed(os.path.join(output_dir, f"{key}.npz"), data=data)
        keys.append(key)
    return prefix, keys


def main():
    parser = argparse.ArgumentParser(description="Kaidee Compiler - pre-process visual assets")
    parser.add_argument("--source", nargs="+", default=["input/source"], metavar="PATH",
                        help="Source code files or directories. Default: input/source/")
    parser.add_argument("--images", default=IMAGE_GLOB, metavar="GLOB",
                        help=f"Image glob pattern (default: {IMAGE_GLOB})")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image compilation (source code only)")
    parser.add_argument("--output", default=OUTPUT_DIR, metavar="DIR",
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--resolution", default="1920x1080", metavar="WxH",
                        help="Output resolution (default: 1920x1080)")
    args = parser.parse_args()

    global W, H, RESOLUTION
    try:
        w_str, h_str = args.resolution.lower().split("x")
        RESOLUTION = (int(w_str), int(h_str))
    except ValueError:
        print(f"ERROR: Invalid resolution format '{args.resolution}'. Use WxH (e.g. 1920x1080)")
        sys.exit(1)
    W, H = RESOLUTION

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    images = []
    if not args.no_images:
        print("Loading images...")
        images = load_images(args.images)
        if not images:
            return
    else:
        print("Skipping images (--no-images)")

    print("\nCollecting source files...")
    source_files = collect_source_files(args.source)
    print(f"  Found {len(source_files)} source files")

    print("\nAnalyzing source code...")
    code = analyze_code(source_files)
    print(f"  {code}")

    # Render source code as images
    print("\nRendering source code images...")
    source_texts, source_names = [], []
    for f in source_files:
        try:
            with open(f) as fh:
                source_texts.append(fh.read())
            source_names.append(os.path.basename(f))
        except (UnicodeDecodeError, OSError):
            pass

    code_images = []
    if source_texts:
        families_seen = {}
        selected = []
        for i, name in enumerate(source_names):
            fam = _detect_lang_family(name)
            if fam not in families_seen:
                families_seen[fam] = i
                selected.append(i)
                if len(selected) >= 3:
                    break
        for i in range(len(source_names)):
            if i not in selected and len(selected) < 3:
                selected.append(i)

        code_img_idx = 0
        for file_idx in selected:
            text = source_texts[file_idx]
            name = source_names[file_idx]
            family = _detect_lang_family(name)
            palette = _get_palette(family)
            keywords = _get_keywords(family)
            remaining = _MAX_CODE_IMAGES - code_img_idx
            if remaining <= 0:
                break
            for style in _STYLES[:min(len(_STYLES), remaining)]:
                render_fn = _STYLE_FUNCS[style]
                rendered = render_fn(text, name, palette, keywords)
                key = f"code_{code_img_idx}_{style}"
                code_images.append((key, rendered))
                code_img_idx += 1
                print(f"  {key}: {name} ({family})")
        print(f"  Generated {len(code_images)} code images")
    else:
        print("  No source code to render")

    keys = []

    # Compile images + code images in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    workers = min(multiprocessing.cpu_count(), len(images) + len(code_images), 8)

    all_jobs = []
    for i, img in enumerate(images):
        all_jobs.append((f"img{i}", img, output_dir))
    for name, img in code_images:
        all_jobs.append((name, img, output_dir))

    _CODE_VARIANTS = [
        "sharp", "abstract", "deep", "sorted",
        "radial_30", "radial_60", "radial_90",
        "fluid_40", "fluid_80", "fluid_120",
    ]

    total = len(all_jobs)
    print(f"\nCompiling {total} images using {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compile_image, job): job[0] for job in all_jobs}
        done = 0
        for future in as_completed(futures):
            prefix, image_keys = future.result()
            keys.extend(image_keys)
            done += 1
            print(f"  [{done}/{total}] {prefix} ({len(image_keys)} variants)")

    # Vignette mask
    vig = make_vignette(H, W)
    np.savez_compressed(os.path.join(output_dir, "vignette.npz"), data=vig)

    # Metadata
    meta = {
        "resolution": list(RESOLUTION),
        "n_images": len(images),
        "code_images": [name for name, _ in code_images],
        "code_variants": _CODE_VARIANTS,
        "code_features": code,
        "frame_keys": keys,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, fn))
        for fn in os.listdir(output_dir)
    )
    print(f"\nDone! {len(keys)} frames -> {output_dir}/ ({total_bytes / 1024 / 1024:.0f} MB)")


if __name__ == "__main__":
    main()
