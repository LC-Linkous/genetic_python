#! /usr/bin/python3

##--------------------------------------------------------------------\
#   genetic_python
#   './src/obj_func_img/shape_decoder.py'
#   PURE helper shared by the headless objective and the optional GUI
#   renderer. Converts a flat array of active genes into a list of
#   drawable shape descriptors. No optimizer state, no GUI imports, so
#   both the scoring path and the display path decode identically and
#   can never drift apart.
#
#   Gene feature layout (matches the original GA_Drawing_Example):
#     circle    (shape 0, 6 feats): x, y, radius, h, s, l
#     square    (shape 1, 5 feats): x, y, h, s, l   (side is fixed)
#     rectangle (shape 2, 7 feats): x, y, w, h_size, h, s, l
#   All feature values are in [0,1]. Position/size scale to canvas here.
#
#   Author(s): Lauren Linkous
#   Last update: June 6, 2026
##--------------------------------------------------------------------\

import numpy as np

try:
    from colour import Color
    _HAVE_COLOUR = True
except Exception:
    _HAVE_COLOUR = False


def _hsl_to_rgb(hsl):
    # returns an (r,g,b) tuple of 0-255 ints
    h, s, l = [float(np.clip(v, 0, 1)) for v in hsl]
    if _HAVE_COLOUR:
        rgb = Color(hsl=(h, s, l)).rgb
        return tuple(int(255 * c) for c in rgb)
    # fallback HSL->RGB if the 'colour' package isn't installed
    def hue(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1 / 6: return p + (q - p) * 6 * t
        if t < 1 / 2: return q
        if t < 2 / 3: return p + (q - p) * (2 / 3 - t) * 6
        return p
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue(p, q, h + 1 / 3)
        g = hue(p, q, h)
        b = hue(p, q, h - 1 / 3)
    return (int(255 * r), int(255 * g), int(255 * b))


def decode_genes(active_genes, num_features, shape_type,
                 canvas_w, canvas_h, scale=0.3, min_size=0.01,
                 square_length=0.1):
    """
    active_genes : 1D flat array (n_active * num_features,) OR 2D (n_active, num_features)
    returns      : list of dicts describing each shape, ready to draw:
                   {'type','x','y','w','h','rgb'}  (w/h meaning depends on type)
    """
    genes = np.array(active_genes).reshape(-1, num_features)
    shapes = []
    for gene in genes:
        x = float(gene[0])
        y = float(gene[1])
        hsl = gene[-3:]
        draw_x = int(np.clip(x, 0, 1) * canvas_w)
        draw_y = int(np.clip(y, 0, 1) * canvas_h)

        remaining = gene[2:-3]  # strip x,y and h,s,l
        rgb = _hsl_to_rgb(hsl)

        if len(remaining) == 1:          # circle: radius
            r = int((float(remaining[0]) * scale + min_size) * canvas_w)
            shapes.append({'type': 'circle', 'x': draw_x, 'y': draw_y,
                           'w': max(r, 1), 'h': max(r, 1), 'rgb': rgb})
        elif len(remaining) == 0:        # square: fixed side
            l = int((square_length * scale + min_size) * canvas_w)
            shapes.append({'type': 'square', 'x': draw_x, 'y': draw_y,
                           'w': max(l, 1), 'h': max(l, 1), 'rgb': rgb})
        else:                            # rectangle: width, height
            w = int((float(remaining[0]) * scale + min_size) * canvas_w)
            h = int((float(remaining[1]) * scale + min_size) * canvas_w)
            shapes.append({'type': 'rectangle', 'x': draw_x, 'y': draw_y,
                           'w': max(w, 1), 'h': max(h, 1), 'rgb': rgb})
    return shapes
