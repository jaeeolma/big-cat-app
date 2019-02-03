# Which of the big cats is in the image

Simple web application for image classification. Input an image, and model tries to predict which of the species of the extended big cat family. Possible labels are

- Cheetah 
- Clouded leopard 
- Cougar 
- Jaguar 
- Leopard
- Lion
- Snow leopard
- Sunda clouded leopard 
- Tiger

Network used here is retrained resnet50, refitted with self-collected dataset of big cats (approximately 3000 images, about 150-370 samples for each species). Network training is done with fast.ai. 

Adapted from https://github.com/pankymathur/google-app-engine
