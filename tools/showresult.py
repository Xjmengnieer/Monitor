import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from monitor_model import bulid_classifier
from utils import *
from datasets.init_train_cate import init_cates, id2dirname

def classify_image(image):
    config = config_load(args.config)

    if config.inference.load_from:
        model = torch.load(config.inference.load_from, map_location='cpu').eval().cuda()
    else:
        model = bulid_classifier(config).eval().cuda()

    trans = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.datasets.mean, std=config.datasets.std)
    ])

    image = Image.fromarray(image).convert('RGB') # 将gradio传入的image转换为PIL.Image
    image = trans(image).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(image)

    pred = torch.sigmoid(output)
    pred_ = (pred > 0.5).long()

    index = torch.where(pred_ == 1) 
    pred_classes = index[1]
    
    confidence_scores = pred[0][pred_classes]

    labels_with_names = [
        f"{id2dirname[str(label.item())]} ({label.item()}) - Confidence: {confidence.item():.4f}"
        for label, confidence in zip(pred_classes, confidence_scores)
    ]

    return "\n".join(labels_with_names)
    # labels_with_names = [f"{id2dirname[str(label.item())]} ({label.item()})" for label in pred_classes]
    # # return "Predicted Labels: " + ", ".join([str(label.item()) for label in pred_classes])
    # return "Predicted Labels: " + ", ".join(labels_with_names)

def main(args):
    interface = gr.Interface(fn=classify_image, inputs="image", outputs="text", live=True)
    interface.launch(share=False)

if __name__ == '__main__':
    args = get_args()
    main(args)
