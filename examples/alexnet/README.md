This example uses the pytorch version of AlexNet as defined here:
https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

It just does testing, showing activations only.  The images are just some random items from the ImageNet corpus, in the standard training set, but it should be easy to see how to change it to test whatever images you want.

It is a good idea to change the netview display range at the upper right, to turn off the `ZeroCtr` flag, and set the `Max` value to 10, to display more of the full range of activation values.

You can pan and zoom interactively to see more details about how the model is coding things in the different layers.  Because every layer has a topographic (retinotopic) mapping onto the image, it is easy to visually see what kinds of features it is picking up on.


