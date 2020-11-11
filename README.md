#Digit Recognizer

Digit Recognizer provides you an ability to draw a digit and get its predicted value from a Neural Net 
model with a very simple interface.

Run run.py and draw the ugliest digit you can and the model with recognize it with 99.5% accuracy!

## Example

Drawn digit:

<img src="/home/ilolio/PycharmProjects/Digit_Recognizer/README_Media/intro.png" alt="drawing" width="200"/>

Prediction:

<img src="/home/ilolio/PycharmProjects/Digit_Recognizer/README_Media/intro2.jpg" alt="drawing"/>
<br/>

## Summary
* Accuracy of the model - 99.5%
* Amount of data to train - ~4000 samples (Compared to 60000 samples in MNIST)
* Model architecture - CNN
* For graphics pygame library is used

## Technical Detailes
The model is a CNN trained on only 3000 digits from MNIST dataset and additional ~1000 digits drawn by me. 
The model building and training performs in src/modeling/model.py.

The additional digits are needed because pure MNIST doesn't serve as a good example of digits drawn using 
computer mouse/touchpad. Hence, handmade data was necessary to build a digit recognizer with sufficient 
accuracy. Handmade data was built using src/data/expand_dataset.py module.

To get an input image of a digit from a user, pygame library was used. run.py implements all the pygame 
part and after the user draws a digit run.py calls prediction.py to get the prediction from the model.