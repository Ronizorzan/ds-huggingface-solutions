from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests
import gradio as gd



processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


def detect_objects(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]


    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
        draw.rectangle(box, outline="red", width=1)
        draw.text((box[0], box[1]), label_name, fill="white" )
    return image, f"Detected {len(results['scores'])} objects: " + ", ".join([model.config.id2label[label.item()] for label in results["labels"]])


app = gd.Interface(
    title="Object Detection with DETR",
    description="Upload an image to detect objects using the DETR model.",
    fn=detect_objects,
    inputs=gd.Image(type="pil", label="Upload Image", width=400, height=400),
    outputs=[gd.Image(type="pil", label="Detected Objects", width=400, height=400),
             gd.Markdown(label="Detection Details")]

)

if __name__ == "__main__":
    app.launch(share=False)


