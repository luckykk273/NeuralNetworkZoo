## Preface
####Hello, before you browse this repository, there are some things have to know:
1. All implementations in this repository are only for practicing(non-commercial).
2. I can't guarantee that all code I wrote in this repository are correct.  
   But I tried to simply express the concept in the annotations when a new function(or concept is used).  
   If there is any mistake about my code(or annotations), welcome to contact me for discussion!
3. Every model in this repository doesn't tune any hyperparameters,  
   so the result of prediction(or reconstruction) cannot totally represent a model's good or bad.
4. Highly recommend follow the order in the reference [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/).  
   Because I implement the code from basics to complexity(and discuss more about the neural network) in order.
5. In reference, I only provide an URL about all structures of networks.  
I didn't cite(or fork) the source code I referred to. The reason is: 
   - It last for a long time to finish all structures, so I can only remember where some sources come from.
   - There are too many references has to be cited.
    
   I feel really sorry about that, so instead, I write down the references I remembered in docstrings on the top of that 'network.py' file.  
   If someone finds any code without citation in this repository is much like yours(and if you mind),  
   please contact me, I will take it off or add the citation as soon as possible.

## Purpose
#### There are three main purposes as follows:  
1. Practice implementing different structures of neural networks mainly with Tensorflow.  
   (Note that some of them will contain numpy version.)  
   
2. Learn how a network works in (mathematical)theory and convert it to code.  

3. Because of my interest of image recognition(both supervised and unsupervised methods),  
   all examples of networks use MNSIT data set as training data.
   - Supervised learning: image classification
   - Unsupervised learning: image generation(or said reconstruction)

## Usage
There is a demo() function in every `[network].py`.  
First import the `[network].py` from package `Networks`:
```python
from Networks import SingleLayerPerceptron  
```
And then just simply call it in `main.py`.  
e.g. Single Layer Perceptron:
```python
if __name__ == '__main__':  
    SingleLayerPerceptron.demo()  
```

## Reference
All structures of neural networks come from the URL below:  
[THE NEURAL NETWORK ZOO](https://www.asimovinstitute.org/neural-network-zoo/),  
POSTED ON SEPTEMBER 14, 2016 BY FJODOR VAN VEEN

## Contact
Welcome to contact me for any further question, below is my gmail address:
* luckykk273@gmail.com

## Others
There are some applications can be implemented but without details.  
If you have interest of them, just contact me and I will fulfill them as soon as possible.  

Review my other repositories with different topics: 
- [Neural Network](): Network models written in tensorflow and numpy.
- [Computer Science](): Data structures and algorithms.
- [Computer Vision](): 2D/3D vision algorithms.
