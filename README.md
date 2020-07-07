# Digit-Detection
<h1> About </h1>
<ul> 
<li> This is part of the offline Text Transcription Project </li>
<li> This project explores text recognition through the usage of Convolutional Nueral Networks, trying to figure out which
model would be the most accurate</li>
<li> Model 1 was examined text recognition using two convolutional2D, along one pooling, flattening and dense layer. </li>
<li> Model 2 explores the idea that sometimes a bigger iterative pattern matching triangle could produce a more accurate
model. This Model is like Model 1 with an additional dense layer, rearranged so that it is pooled after each convolutional step</li>
<li> Both the EMNIST Letter's and Number's data set is used to Test/Train this model
</ul>

<h1> Libraries </h1>
<ul> 
<li> Keras </li>
<li> EMNIST </li>
<li> Tensorflow </li>
</ul>

<h1> Results and Notes </h1>
<ul>
<li> Build a two models that were capable of reconizing the test set 92.4% and 93.4% of the time respectable </li>
<li> Model 2 is much more accurate then Model 1 </li>
<li> For the project Model 2 will be used for the CNN Layer </li>
