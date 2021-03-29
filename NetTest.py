import APS360_NetTrainAcc
from ipynb.fs.full.DatasetSplit import get_data_loader_type
from torchvision import transforms
import torch
import numpy as np

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

    best_test_acc = -1
    best_lr = 0.0001
    for lr in np.arange(0.0001, 0.0010, 0.00005):
        print("Testing Learning Rate: {}".format(lr))
        APS360_NetTrainAcc.train_net(model, train, val, batch_size=batch_size, num_epochs=10, learning_rate=lr)
        test_acc = APS360_NetTrainAcc.accuracy_net(model, test)
        
        # Print Test Accuracy
        print("Test Accuracy = {:%}".format(test_acc))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_lr = lr
    # Print Test Accuracy
    print("Best Test Accuracy = {:%}".format(best_test_acc))
    print("Best LR = {}".format(best_lr))

    # Best LR = 0.0003