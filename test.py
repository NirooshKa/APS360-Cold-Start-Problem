def train(model, batch_size = 120, learning_rate = 0.001, num_epochs = 1):

  #Step 1: Setup
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
  iters, losses, train_acc, val_acc = [], [], [], []

  #Step 2: Training
  epochs = 0
  for n in range(num_epochs):
    for images, labels in iter(trainLoader):

      #For enabling CUDA core
      if use_cuda and torch.cuda.is_available():
        images = imgs.cuda()
        labels = labels.cuda()

      out = model(images) #Forward pass
      loss = criterion(out, labels) #Compute total losses
      loss.backward() #Backward pass 
      optimizer.step() #Update each parameter
      optimizer.zero_grad() #Clean up step for Pytorch
      epochs += 1

    #Let's save the train + validation information
    iters.append(n)
    losses.append(float(loss)/batch_size) #Average loss
    train_acc.append(get_accuracy(model,train = True))
    val_acc.append(get_accuracy(model,train = False))
    path = "model_{0}_bs{1}_lr{2}_epochs{3}".format(model.name, batch_size, learning_rate, n)
    torch.save(model.state_dict(), path)

    print("Epoch #:", n+1, "Train acc:", train_acc[n], "Val acc:", val_acc[n])

    #Step 3: Plot Curves
  plt.title("Training curve")
  plt.plot(iters, losses, label = "Train")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.show()

  plt.title("Training curve")
  plt.plot(iters, train_acc, label = "Train")
  plt.plot(iters, val_acc, label = "Validation")
  plt.xlabel("Iterations")
  plt.ylabel("Training Accuracy")
  plt.legend(loc = 'best')
  plt.show()

  print("Final Training Accuracy: {}".format(train_acc[-1]))
  print("Final Validation Accuracy: {}".format(val_acc[-1]))

  return train_acc, val_acc, epochs

def get_accuracy(model, train = False):
  
  correct = 0
  total = 0

  if train:
    dataLoader = trainLoader
  else:
    dataLoader = valLoader

  for images, labels in dataLoader:

    #For enabling CUDA core
    if use_cuda and torch.cuda.is_available():
      images = images.cuda()
      labels = labels.cuda()

    output = model(images)
    pred = output.max(1, keepdim = True)[1] #Index w/max prediction score
    correct += pred.eq(labels.view_as(pred)).sum().item()
    total += images.shape[0]
  
  return correct/total

