# Keras sequence to sequence example
Generic Keras implementation of a sequence to sequence model with several examples.

[![license](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/philipperemy/keras-seq2seq-example/blob/master/LICENSE) [![dep1](https://img.shields.io/badge/Tensorflow-1.2+-blue.svg)](https://www.tensorflow.org/) [![dep2](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) 

<p align="center">
  <img src="http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png" width="700">
  <br><i>Encoder Decoder model (seq2seq)</i>
</p>




# Usage

```
git clone https://github.com/philipperemy/keras-seq2seq-example.git
cd keras-seq2seq-example
python3 utils.py # build the vocabulary and the characters.
export CUDA_VISIBLE_DEVICES=0; nohup python3 -u model.py &
```
