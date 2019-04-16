# kaggle-cats-and-dogs
Run ResNet18 to perform binary classification on the [cats-and-dogs dataset](https://www.kaggle.com/tongpython/cat-and-dog) from Kaggle. The trained model is used to generate a saliency map which represents the "implicit attention" of the CNN. The details of how to generate saliency map is introduced in the paper [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034). To visualize the saliency map to make it more explainatory to human, it is smoothed by Gaussian filter so that it looks like a "heatmap". Below is an example of an image with its saliency map and heatmap:
![alt text](https://github.com/wentaoveggiebird/kaggle-cats-and-dogs/blob/master/images/dog1215.png)

