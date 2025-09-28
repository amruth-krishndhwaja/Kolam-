import argparse
from extract import extract_hex_from_image
from utils import heuristic_symmetry_classifier, ring_around_dots, draw_curves_to_image, image_to_base64
from PIL import Image

def demo(img_path, out_path):
    out = extract_hex_from_image(img_path, debug=False)
    print("HEX:", out['hex'])
    print("Matrix shape:", out['binary_matrix'].shape)
    label = heuristic_symmetry_classifier(out['binary_matrix'])
    print("Heuristic classification:", label)
    curves = ring_around_dots(out['binary_matrix'], spacing=1.0, radius=0.25, n_circle_pts=32)
    img = draw_curves_to_image(curves, show_grid=(out['binary_matrix'].shape[0], out['binary_matrix'].shape[1], 1.0))
    img.save(out_path)
    print("Saved generated kolam to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", default="demo_output.png")
    args = parser.parse_args()
    demo(args.img, args.out)
