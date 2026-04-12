import gradio as gr
from PIL import Image
from transformers import pipeline

# Your existing code
pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

def detect_objects(image):
    # The pipeline handles the detection
    results = pipe(image)
    # You can then process 'results' to draw boxes or return text
    return results

demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs="json"
)

demo.launch()
