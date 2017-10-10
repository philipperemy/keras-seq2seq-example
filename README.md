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

# Example

Based on a Japanese postal address, predict the corresponding ZIP Code.

This address `福島県会津若松市栄町２−４` corresponds to `965-0871`.

The current data set is composed of postal addresses, scraped from the Japanese yellow pages [itp.ne.jp](itp.ne.jp). One line looks like this:

<p align="center">
  <img src="assets/IMG_1.png" width="400">
  <br><i>Row of the data set</i>
</p>

We extract the left part (target) and the right part (inputs) and we build a supervised learning problem.

We expect the accuracy to be very high because there is a lot of redundancy in the addresses.

Let's also mention that Google contains a big database and doing some lookups are possible. It should give a nearly perfect accuracy.

The question is: Why do we bother building this model?

- For the sake of learning!

- Google does not deal with unseen addresses (change a number by another one in an address, and see if Google knows about it).
- If one or more characters are missing, Google hardly handles it. Deep learning can still make a prediction. 
- We can add noise in the addresses (such as Dropout or character replacement) and train a model on this augmented data set.
- Also it works totally offline (nowadays, it's less important though!)

<p align="center">
  <img src="assets/IMG_0162.jpg" width="400">
  <br><i>Encoder Decoder model (seq2seq)</i>
</p>

```
Iteration 1
Train on 382617 samples, validate on 42513 samples
Epoch 1/10
382617/382617 [==============================] - 216s - loss: 0.8973 - acc: 0.6880 - val_loss: 0.3011 - val_acc: 0.8997
Epoch 2/10
382617/382617 [==============================] - 197s - loss: 0.1868 - acc: 0.9401 - val_loss: 0.1296 - val_acc: 0.9589
Epoch 3/10
382617/382617 [==============================] - 196s - loss: 0.0921 - acc: 0.9718 - val_loss: 0.0790 - val_acc: 0.9763
Epoch 4/10
382617/382617 [==============================] - 200s - loss: 0.0586 - acc: 0.9825 - val_loss: 0.0562 - val_acc: 0.9839
Epoch 5/10
382617/382617 [==============================] - 201s - loss: 0.0440 - acc: 0.9871 - val_loss: 0.0535 - val_acc: 0.9848
Epoch 6/10
382617/382617 [==============================] - 197s - loss: 0.0345 - acc: 0.9900 - val_loss: 0.0334 - val_acc: 0.9908
Epoch 7/10
382617/382617 [==============================] - 198s - loss: 0.0279 - acc: 0.9920 - val_loss: 0.0305 - val_acc: 0.9918
Epoch 8/10
382617/382617 [==============================] - 196s - loss: 0.0239 - acc: 0.9932 - val_loss: 0.0234 - val_acc: 0.9938
Epoch 9/10
382617/382617 [==============================] - 199s - loss: 0.0207 - acc: 0.9942 - val_loss: 0.0253 - val_acc: 0.9935
Epoch 10/10
382617/382617 [==============================] - 200s - loss: 0.0180 - acc: 0.9950 - val_loss: 0.0263 - val_acc: 0.9933
Q -------------------福島県会津若松市栄町２−４
T 965-0871
☑ 965-0871
---
Q -----------------東京都品川区西品川３丁目５−４
T 141-0033
☑ 141-0033
---
Q -------------------滋賀県愛知郡愛荘町市１５７
T 529-1313
☑ 529-1313
---
Q ----------------青森県つがる市木造赤根１３−４０
T 038-3142
☑ 038-3142
---
Q ---------------大阪府東大阪市中鴻池町１丁目６−６
T 578-0975
☑ 578-0975
---
Q ------------------東京都千代田区一番町２７−４
T 102-0082
☑ 102-0082
---
Q ------------神奈川県横須賀市太田和４丁目２５５０−１
T 238-0311
☑ 238-0311
---
Q ------------鹿児島県南さつま市笠沙町片浦２３４７−６
T 897-1301
☑ 897-1301
---
Q ---------------千葉県東金市田間１１５−１−１０２
T 283-0005
☑ 283-0005
---
Q ---------------千葉県匝瑳市八日市場イ２４０４−１
T 289-2144
☑ 289-2144
```
