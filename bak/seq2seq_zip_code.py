import matplotlib.pyplot as plt
import tensorflow as tf

import helpers
from load_data import get_partial, get_batch, VOCAB_SIZE

x, y = get_partial()
xt, xlen = helpers.batch(x)

vocab_size = VOCAB_SIZE

print(x)
print(xt)
print(xlen)

tf.reset_default_graph()
sess = tf.InteractiveSession()

print(tf.__version__)

PAD = 0
EOS = 1

# vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

# decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)

# del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, encoder_outputs,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)
# decoder_targets int32 tensor is shaped [decoder_max_time, batch_size]
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

#
# batch_ = [[6], [3, 4], [9, 8, 7]]
#
# batch_, batch_length_ = helpers.batch(batch_)
# print('batch_encoded:\n' + str(batch_))
#
# din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
#                             max_sequence_length=4)
# print('decoder inputs:\n' + str(din_))
#
# pred_ = sess.run(decoder_prediction,
#                  feed_dict={
#                      encoder_inputs: batch_,
#                      # decoder_inputs: din_,
#                  })
# print('decoder predictions:\n' + str(pred_))
#
batch_size = 2


def next_feed():
    x, y = get_batch()
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(y)
    # decoder_inputs_, _ = helpers.batch(
    #     [[EOS] + (sequence) for sequence in batch]
    # )
    return {
        encoder_inputs: encoder_inputs_,
        # decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }


loss_track = []

max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')

plt.plot(loss_track)
plt.show()
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track) * batch_size, batch_size))
