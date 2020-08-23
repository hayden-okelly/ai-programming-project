from utilities import *

epochs = args.epochs
steps = 0
print_every = 40
model.to(device)

for e in range(epochs):
    model.train()
    running_loss = 0
    
    for images, labels in iter(trainloader):
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            
            model.train()

model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epochs': epochs,
              'model': getattr(models, args.arch)(pretrained=True),
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

torch.save(checkpoint, args.save_dir)