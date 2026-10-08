import numpy as np

def describe_region(heat, thr=0.6):
    """heat: 2D Grad-CAM array in [0,1]."""
    if heat.max() <= 0:
        return None
    h, w = heat.shape
    mask = heat >= thr * heat.max()
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / h, xs.mean() / w
    left_share = (xs < w / 2).mean()
    # PA chest film: the patient's RIGHT lung appears on the LEFT of the image
    side = ("both lungs" if 0.3 < left_share < 0.7
            else "right lung" if left_share >= 0.7 else "left lung")
    zone = "upper" if cy < 1/3 else "middle" if cy < 2/3 else "lower"
    edge = cx < 0.08 or cx > 0.92 or cy < 0.05 or cy > 0.95
    return {"text": f"{zone} part of the {side}", "unreliable": edge}