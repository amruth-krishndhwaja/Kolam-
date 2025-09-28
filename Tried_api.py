from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import io, os
from PIL import Image
import base64
from extract import extract_hex_from_image
from utils import image_to_base64, load_image_to_tensor, tensor_to_pil, heuristic_symmetry_classifier, ring_around_dots, draw_curves_to_image
import torch
from models.encoder import KolamEncoder
from models.generator import CondGenerator

app = FastAPI(title="Kolam Prototype API")

ENCODER_PATH = "models/encoder.pth"
GENERATOR_PATH = "models/generator.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_model = None
generator_model = None

if os.path.exists(ENCODER_PATH):
    try:
        encoder_model = KolamEncoder(pretrained=False)
        encoder_model.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
        encoder_model.to(device).eval()
        print("Loaded encoder weights.")
    except Exception as e:
        print("Failed to load encoder:", e)
if os.path.exists(GENERATOR_PATH):
    try:
        generator_model = CondGenerator().to(device)
        generator_model.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))
        generator_model.eval()
        print("Loaded generator weights.")
    except Exception as e:
        print("Failed to load generator:", e)

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read()
    tmp = "/tmp/upload_extract.jpg"
    open(tmp,"wb").write(data)
    out = extract_hex_from_image(tmp, debug=False)
    return JSONResponse({"hex": out['hex'], "matrix_shape": out['binary_matrix'].shape})

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    data = await file.read()
    tmp = "/tmp/upload_classify.jpg"
    open(tmp,"wb").write(data)
    try:
        out = extract_hex_from_image(tmp, debug=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    bin_mat = out['binary_matrix']
    if encoder_model is not None:
        img_tensor = load_image_to_tensor(tmp, size=384, device=device)
        with torch.no_grad():
            logits, cls = encoder_model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()[0]
            label_idx = int(torch.argmax(logits, dim=1).cpu().item())
            return {"label_index": label_idx, "probs": probs, "hex": out['hex']}
    else:
        h = heuristic_symmetry_classifier(bin_mat)
        return {"label": h['label'], "score": h['score'], "hex": out['hex']}

@app.post("/generate")
async def generate(file: UploadFile = File(...), n: int = Form(3)):
    data = await file.read()
    tmp = "/tmp/upload_gen.jpg"
    open(tmp,"wb").write(data)
    try:
        out = extract_hex_from_image(tmp, debug=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    bin_mat = out['binary_matrix']
    rows, cols = bin_mat.shape
    # if generator model exists use it
    if generator_model is not None and encoder_model is not None:
        img_tensor = load_image_to_tensor(tmp, size=128, device=device)
        with torch.no_grad():
            _, cls = encoder_model(img_tensor)
            imgs_b64 = []
            for i in range(n):
                z = torch.randn(1, 128).to(device)
                fake = generator_model(cls, z)
                pil = tensor_to_pil(fake)
                imgs_b64.append(image_to_base64(pil))
            return {"images": imgs_b64, "hex": out['hex']}
    # fallback renderer
    curves = ring_around_dots(bin_mat, spacing=1.0, radius=0.25, n_circle_pts=32)
    img = draw_curves_to_image(curves, show_grid=(rows,cols,1.0))
    return {"images": [image_to_base64(img)], "hex": out['hex']}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
