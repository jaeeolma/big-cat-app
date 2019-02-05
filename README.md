# Which of the wild cats is in the image

Simple web application for image classification. Input an image, and model tries to predict which of the species of the extended big cat family. Possible labels are

- Caracal
- Cheetah 
- Clouded leopard 
- Cougar 
- Jaguar
- Jungle cat
- Leopard
- Lion
- Lynx
- Serval
- Snow leopard
- Sunda clouded leopard 
- Tiger

Network used here is retrained resnet50, refitted with self-collected dataset of big cats (approximately 3000 images, about 150-370 samples for each species). Network training is done with fast.ai. 

Adapted from https://github.com/pankymathur/google-app-engine
