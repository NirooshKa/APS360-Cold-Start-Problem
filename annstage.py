import APS360_NetTrainAcc
from ipynb.fs.full.DatasetSplit import get_data_loader_type
from torchvision import transforms
import torch
import sys
from PIL import Image, ImageFile

# torch.manual_seed(1) # for reproducibility
ImageFile.LOAD_TRUNCATED_IMAGES = True

def detect_using_model(subject, source_image):
    classes = {"cats": 67, "cats_and_dogs": 37, "horses": 7, "flowers": 5}
    weights = {"cats": "cats.pt", "cats_and_dogs": "cats_and_dogs.pt", "horses": "horses.pt", "flowers": "flowers.pt"}
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

    pretrained = APS360_NetTrainAcc.alexnet_init()
    model = APS360_NetTrainAcc.ALFinalClassifier(classes[subject], pretrained)
    model.load_state_dict(torch.load(weights[subject]))
    model.eval()

    image = preprocess(Image.open(source_image).load())
    batch = torch.unsqueeze(image, 0) # Manual batching
    out = model(batch)

    print(out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Not enough arguments.")
    else:
        detect_using_model(sys.argv[1], sys.argv[2])