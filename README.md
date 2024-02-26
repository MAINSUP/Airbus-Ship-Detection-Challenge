# Airbus-Ship-Detection-Challenge-Winstars.AI-
Current repository contains Kaggle Airbus Ship Detection Challenge competition notebook and python script files for both high and low level model training and inference.

Notebook can be directly uploaded to Kaggle and used as it is on Airbus Ship Detection Challenge input dataset.
If one would like to use it locally, paths to photo library and working directory have to be updated.
Notebook code consits of 5 sections.

Section 1 is dealing with images dataset. Here an EDA and image prepocessing are perfomed.
After decoding RLE masks, images (random) are ploted with ground truth bounding boxes to demonstrate correctness of decoding.
Decoded data is also saved in CSV file for ease of further use.

Section 2 of the notebook contains few code lines for training daset preparation.
Since tensorflow framework is used for model training, datasets should be prepared in the form of tensors of a respective ranks.
Training and validation images are given in the form 4th rank tensors, like T = (N, H, W, 3), 
where N is the number of images in the training dataset, H is the image height (pixels), W is the image width (pixels), 3 is the colour channels (RGB).
Since models perform better with normilized inputs, channels are normilized to be in range [-1, 1] or [0, 1] as found necessary.
Training and validation bounding boxes (targets) are given in the form of 4 component vector, like Box = [x1, y1, x2, y2],
where x and y are coordinates in pixels.

Section 3. Since it a common approach to utilize transfer learning while developing ML models, section 3 contains an example of it.
A pretrained model " " is used to build training model that consits of dense, dropout layers.
Hyperparameters are number of dense layers, dropout coeficient, learning rate.
Callbacks are used to watch training performance and visualize it with use of Tensorboard libriary as found necessary.

In Section 4 of the notebook, a custom training model is build to compare performance of pretrained model from Sec. 3 and provide few extra hypermarameters from image augmentation layer.
It consists of dense and dropout layers as well.

Finally, Section 5 as intended for model performance evaluation.
In order to calculate IoU values predictions are made on the same dataset that was used for training, as Kaggle competetion test dataset does not contain ground truth data.
The model structure and training performance is visualized with Tensorboard widget.

Further model inference is based on TFLite model, a compact model format, optimized for deployment. 
Few lines of code demonstrate, how to save Tensorflow model and the convert it to TFLite format.

Python files .py and .py can be used for model training and model inference, respectively.
To run, input files should be located in the same directory as script files.

✔Please check requirements.txt for required libriaries for running the code.



👍 Feel free to use deposited code in your custom projects.
