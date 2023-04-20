import torch 
import numpy as np 



def train(model, trainloader, valloader, testloader,num_epoch, device, optimizer, criterion):  # Train the model
    print("Start training...")
    trn_loss_hist = []
    trn_acc_hist = []
    val_acc_hist = []
    test_acc_hist = []
    model.train()  # Set the model to training mode

    best_val_loss, best_val_loss_epoch = -100, 0
    best_test_acc = 0 
    for i in range(num_epoch):
        running_loss = []
        print('-----------------Epoch = %d-----------------' % (i+1))
        for image, label in trainloader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()  
            pred = model(image)
            # print("pred: ", pred.shape)
            # print("label: ", label.shape)
            loss = criterion(pred, label)  
            # print(loss)
            running_loss.append(loss.item())
            loss.backward()  
            optimizer.step() 
        print("\n Epoch {} loss:{}".format(i+1, np.mean(running_loss)))

        # Keep track of training loss, accuracy, and validation loss
        trn_loss_hist.append(np.mean(running_loss))
        train_acc, train_loss = evaluate(model, trainloader, criterion=criterion, device=device)
        val_acc, val_loss = evaluate(model, valloader, criterion=criterion, device=device)
        test_acc, test_loss = evaluate(model, testloader, criterion=criterion, device=device)

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = i 
            best_test_acc = test_acc
        trn_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        test_acc_hist.append(test_acc)
        print('Epoch: {:02d}, trloss: {:.4f}, tracc: {:.4f}, valloss: {:.4f}, valacc: {:.4f}, testloss: {:.4f}, testacc: {:.4f}'.format(i,train_loss,train_acc, val_loss, val_acc, test_loss, test_acc))
    print(f"Result Test Accuracy According to smallest validation loss: {round(best_test_acc * 100, 2)}")
    # print("Done!")
    return trn_loss_hist, trn_acc_hist, val_acc_hist


def evaluate(model, loader, criterion, device):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total_loss = 0 
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for image, label in loader:
            image = image.to(device)
            # print(image)
            label = label.to(device)
            pred = model(image)
            curr_loss = criterion(pred, label)
            total_loss += curr_loss 
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
        acc = correct/len(loader.dataset)
        loss = total_loss / len(loader.dataset)
        # print("\n Evaluation accuracy: {}".format(acc))
        return acc, loss 


