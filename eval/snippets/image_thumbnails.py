"""Generate product image thumbnails in the sizes the storefront needs."""
SIZES = {"thumb": 160, "grid": 480, "zoom": 1600}


def fit(width, height, target):
    scale = target / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def plan(images):
    return [
        {"src": img["path"], "variant": name, "size": fit(img["w"], img["h"], px)}
        for img in images
        for name, px in SIZES.items()
        if max(img["w"], img["h"]) > px
    ]
