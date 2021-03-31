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
    labels = {
        "cats": ["Abyssinian", "American Bobtail", "American Curl", "American Shorthair", "American Wirehair", "Applehead Siamese", "Balinese", "Bengal", "Birman", "Bombay", "British Shorthair", "Burmese", "Burmilla", "Calico", "Canadian Hairless", "Chartreux", "Chausie", "Chinchilla", "Cornish Rex", "Cymric", "Devon Rex", "Dilute Calico", "Dilute Tortoiseshell", "Domestic Long Hair", "Domestic Medium Hair", "Domestic Short Hair", "Egyptian Mau", "Exotic Shorthair", "Extra-Toes Cat - Hemingway Polydactyl", "Havana", "Himalayan", "Japanese Bobtail", "Javanese", "Korat", "LaPerm", "Maine Coon", "Manx", "Munchkin", "Nebelung", "Norwegian Forest Cat", "Ocicat", "Oriental Long Hair", "Oriental Short Hair", "Oriental Tabby", "Persian", "Pixiebob", "Ragamuffin", "Ragdoll", "Russian Blue", "Scottish Fold", "Selkirk Rex", "Siamese", "Siberian", "Silver", "Singapura", "Snowshoe", "Somali", "Sphynx - Hairless Cat", "Tabby", "Tiger", "Tonkinese", "Torbie", "Tortoiseshell", "Turkish Angora", "Turkish Van", "Tuxedo", "York Chocolate"],
        "cats_and_dogs": ["Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland", "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier"],
        "horses": ["Akhal-Teke", "Appaloosa", "Arabian", "Friesian", "Orlov Trotter", "Percheron", "Vladimir Heavy Draft"],
        "flowers": ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    }
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

    pretrained = APS360_NetTrainAcc.alexnet_init()
    model = APS360_NetTrainAcc.ALFinalClassifier(classes[subject], pretrained)
    model.load_state_dict(torch.load(weights[subject], map_location=torch.device('cpu')))
    model.eval()

    image = preprocess(Image.open(source_image))
    image = image[:3,:,:] # Remove potential alpha channel
    batch = torch.unsqueeze(image, 0) # Manual batching
    out = model(batch)
    _, index = torch.max(out, 1)
    label = labels[subject][index[0]]

    print(label)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Not enough arguments.")
    else:
        detect_using_model(sys.argv[1], sys.argv[2])