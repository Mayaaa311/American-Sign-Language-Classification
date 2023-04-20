import argparse
from dataset import HandSignDataset
from torchvision import transforms
import os 
from training import train, evaluate
from torch import optim
import torch 
from model import * 
import torchvision
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataset(args):
    # Prepare dataset 

    # Define the transformation(s) to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    ])

    # Call the create_dataset function to create a PyTorch dataset
    test_dataset = HandSignDataset(csv_file='output.csv', root_dir='avg_test', partition='test',transform=transform)
    train_dataset = HandSignDataset(csv_file='output.csv', root_dir='avg_train', partition='train',transform=transform)
    val_dataset = HandSignDataset(csv_file='output.csv', root_dir='avg_dev', partition='dev',transform=transform)
    return train_dataset, val_dataset, test_dataset

def get_model(args):
    if args.model == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        num_features = model.fc.in_features 
        model.fc = nn.Linear(num_features, 4)
    elif args.model == "network":
        model = Network()
    else:
        raise NotImplementedError("Not Implemented model argument")
    
    return model 

def run_exp(args):
    
    train_dataset, val_dataset, test_dataset = prepare_dataset(args) 
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # Training Setup 
    model = get_model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()


    train(model=model, 
          trainloader=trainloader, 
          valloader=valloader, 
          testloader=testloader,
          num_epoch=args.num_epochs, 
          device=device, 
          optimizer=optimizer, 
          criterion=criterion)
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='442 Nicode Project')

    # parser.add_argument('--num_conv_layers', type=int, default=4)
    # parser.add_argument('--hidden_channels', type=int, default=256)
    # parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "network"])

    args = parser.parse_args()
    print(args)
    run_exp(args)