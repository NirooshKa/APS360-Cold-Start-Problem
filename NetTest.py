import APS360_NetTrainAcc
from ipynb.fs.full.DatasetSplit import get_data_loader_type
from torchvision import transforms
import torch

torch.manual_seed(1) # for reproducibility

if __name__ == "__main__":
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])
    batch_size = 64

    train, val, test, classes = get_data_loader_type("horses", batch_size=batch_size, transformer=preprocess)
    print(classes)

    pretrained = APS360_NetTrainAcc.alexnet_init()
    model = APS360_NetTrainAcc.ALFinalClassifier(len(classes), pretrained)
    APS360_NetTrainAcc.train_net(model, train, val, batch_size=batch_size, num_epochs=25, learning_rate=0.0001)

    # Print Test Accuracy
    print("Test Accuracy = {:%}".format(APS360_NetTrainAcc.accuracy_net(model, test)))