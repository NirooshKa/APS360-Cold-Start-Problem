import APS360_NetTrainAcc
from ipynb.fs.full.DatasetSplit import get_data_loader_type
from torchvision import transforms
import torch
from PIL import Image, ImageFile

# torch.manual_seed(1) # for reproducibility
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_once(subject, use_cuda=False):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])
    batch_size = 64

    train, val, test, classes = get_data_loader_type(subject, batch_size=batch_size, num_workers=8, transformer=preprocess)
    print(subject)
    print(classes)

    pretrained = APS360_NetTrainAcc.alexnet_init(use_cuda=use_cuda)
    model = APS360_NetTrainAcc.ALFinalClassifier(len(classes), pretrained)
    if use_cuda:
        model.cuda()
    APS360_NetTrainAcc.train_net(model, train, val, batch_size=batch_size, num_epochs=25, learning_rate=0.0001, use_cuda=use_cuda, subject=subject)

    # Print Test Accuracy
    print("Test Accuracy = {:%}".format(APS360_NetTrainAcc.accuracy_net(model, test, use_cuda=use_cuda)))
    
    # Save trained weights
    torch.save(model.state_dict(), f'{subject}.pt')

if __name__ == "__main__":
    subjects = ["cats"]
    for subject in subjects:
        train_once(subject, use_cuda=False)
