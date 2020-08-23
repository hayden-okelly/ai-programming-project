import train

model, class_to_idx = load_checkpoint('checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
  
    # opens an image using PIL
    
    image = Image.open(image)
    
    # processes a PIL image for use in a PyTorch model
    
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    image = preprocess(image)
    
    return image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Done: Implement the code to predict the class from an image file
    
    img = process_image(image_path)
    model.to(device)
    img = img.to(device)
    
    img_classes_dict = {v: k for k, v in class_to_idx.items()}
    
    model.eval()
    
    with torch.no_grad():
        img.unsqueeze_(0)
        output = model.forward(img)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk)
        probs, classes = probs[0].tolist(), classes[0].tolist()
        
        return_classes = []
        for c in classes:
            return_classes.append(img_class_to_idx[c])
            
        return probs, return_classes

probs, classes = predict(args.image_path, model, args.topk)

print ('Classes: ', return_classes)
print('Probability: ', probs)
