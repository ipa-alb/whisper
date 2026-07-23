"""OWLv2 open-vocabulary detector helper — runs in its OWN venv, not whisper's.

The `_`-prefix keeps the assistant's tool loader from importing it (it needs
torch + transformers, which the dependency-light whisper venv deliberately
lacks). see_camera.py invokes it as a subprocess with the `.venv_owl`
interpreter and reads one JSON line from stdout.

OWLv2 is text-prompted: it detects whatever the queries name, which is what
lets the robot see workshop tools (hammer, mallet, ...) that COCO-trained
YOLO has no class for. CPU inference of the base model takes a few seconds
per frame — fine for a single look() snapshot.

Usage:  .venv_owl/bin/python tools/_owl_detect.py IMAGE THRESHOLD QUERY [QUERY...]
Output: JSON list of {"label": str, "conf": float, "box": [x1, y1, x2, y2]}
        (box in pixels of the input image; label is the bare query text).
First run downloads the model to the HF cache (~600 MB); later runs are local.
"""

import json
import os
import sys


def main():
    image_path, thresh, queries = sys.argv[1], float(sys.argv[2]), sys.argv[3:]

    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = os.environ.get("OWL_MODEL", "google/owlv2-base-patch16-ensemble")
    img = Image.open(image_path).convert("RGB")
    proc = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model.eval()

    # "a hammer" scores noticeably better than bare "hammer" for CLIP-style text
    prompts = [f"a {q}" for q in queries]
    inputs = proc(text=[prompts], images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, threshold=thresh, target_sizes=[img.size[::-1]]
    )[0]

    dets = [
        {
            "label": queries[int(label)],
            "conf": round(float(score), 3),
            "box": [round(float(v), 1) for v in box.tolist()],
        }
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"])
    ]
    print(json.dumps(dets))


if __name__ == "__main__":
    main()
