#! /usr/bin/python3

##--------------------------------------------------------------------\
#   genetic_python
#   './src/obj_func_img/func_F.py'
#   Objective function for the image-drawing problem, written to the SAME
#   contract as the equation objective:  func_F(X, NO_OF_OUTS) -> (F, noError)
#
#   This replaces the standard Objective Function Set equation-based functions
#   with an image comparison. It renders the candidate HEADLESS (Pillow, no GUI) and
#   diffs against a reference image. Swapping this file for the equation
#   func_F, or for a simulation-backed objective that writes/parses an Excel
#   file, requires NO change to genetic_algorithm.py.
#
#   F[0] = mean absolute pixel difference + tiny gene-count penalty  (minimize)
#   This matches the original fitness  -mean(|diff|) - 1e-5*size  with the
#   sign flipped so the suite can minimize toward TARGET 0.
#
#   Author(s): Lauren Linkous
#   Last update: June 6, 2026
##--------------------------------------------------------------------\

import numpy as np
from PIL import Image, ImageDraw

try:
    from shape_decoder import decode_genes
except ImportError:
    from obj_func_img.shape_decoder import decode_genes

# ---- problem binding (set by configs_F.py via set_reference) ----
_REF = {
    'array': None,        # np.array (H, W, 3) uint8 reference image
    'w': None, 'h': None, # canvas dims (default = reference dims)
    'num_features': 5,
    'shape_type': 1,      # 0 circle, 1 square, 2 rectangle
    'scale': 0.3,
    'min_size': 0.01,
    'square_length': 0.1,
    'size_penalty': 1e-5,
}


def set_reference(ref_array, num_features, shape_type,
                  canvas_w=None, canvas_h=None,
                  scale=0.3, min_size=0.01, square_length=0.1,
                  size_penalty=1e-5):
    # called once from the config / driver to bind the problem instance
    ref = np.asarray(ref_array).astype(np.int32)
    h, w = ref.shape[0], ref.shape[1]
    _REF['array'] = ref
    _REF['h'] = canvas_h if canvas_h is not None else h
    _REF['w'] = canvas_w if canvas_w is not None else w
    _REF['num_features'] = int(num_features)
    _REF['shape_type'] = int(shape_type)
    _REF['scale'] = float(scale)
    _REF['min_size'] = float(min_size)
    _REF['square_length'] = float(square_length)
    _REF['size_penalty'] = float(size_penalty)


def render_to_array(active_genes):
    # headless render of the candidate to an (H, W, 3) uint8 array
    w, h = _REF['w'], _REF['h']
    img = Image.new('RGB', (w, h), (255, 255, 255))
    drw = ImageDraw.Draw(img)
    shapes = decode_genes(active_genes, _REF['num_features'], _REF['shape_type'],
                          w, h, scale=_REF['scale'], min_size=_REF['min_size'],
                          square_length=_REF['square_length'])
    for s in shapes:
        x, y, sw, sh, rgb = s['x'], s['y'], s['w'], s['h'], s['rgb']
        if s['type'] == 'circle':
            r = sw
            drw.ellipse([x - r, y - r, x + r, y + r], fill=rgb)
        else:  # square or rectangle
            drw.rectangle([x, y, x + sw, y + sh], fill=rgb)
    return np.asarray(img).astype(np.int32)


def func_F(X, NO_OF_OUTS=1):
    F = np.zeros((NO_OF_OUTS))
    noErrors = True
    try:
        if _REF['array'] is None:
            raise ValueError("reference image not set; call set_reference() first")
        genes = np.array(X).reshape(-1, _REF['num_features'])
        rendered = render_to_array(genes)
        diff = rendered - _REF['array']
        err = np.mean(np.abs(diff)) + _REF['size_penalty'] * genes.size
        F[0] = err
    except Exception:
        noErrors = False
    return F, noErrors
