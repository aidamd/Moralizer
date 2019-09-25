import numpy as np
import tensorflow as tf
from nn import *
import pandas as pd

class GAN():
    def __init__(self, params, domain_df, target_df, vocab):
        self.vocab = vocab
        self.domain = domain_df
        self.target = target_df
        print("Domain sample shape:", len(self.domain))
        print("Target sample shape:", len(self.target))

        self.params = params
        for key in params:
            setattr(self, key, params[key])
        self.embeddings = load_embedding(self.vocab,
                                         "/home/aida/Data/word_embeddings/GloVe/glove.6B.300d.txt",
                                         self.embedding_size)
        self.adversarial_build()
        self.target, self.domain = balance_data(self.target, self.domain)
        batches = get_batches(self.target, self.domain, self.batch_size, vocab.index("<pad>"), vocab.index("<eos>"), vocab.index("<go>"))
        self.adversarial_train(batches)


    def adversarial_build(self):
        tf.reset_default_graph()
        self.keep_prob = tf.placeholder(tf.float32)

        self.X1 = tf.placeholder(tf.int32, [None, None], name="input_1")
        self.X0 = tf.placeholder(tf.int32, [None, None], name="input_0")

        self.maximum_length = 2 * self.X1.shape[1]

        self.decode_X1 = tf.placeholder(tf.int32, [None, None], name="decode_1")
        self.decode_X0 = tf.placeholder(tf.int32, [None, None], name="decode_0")

        self.output_X1 = tf.placeholder(tf.int32, [None, None], name="output1")
        self.output_X0 = tf.placeholder(tf.int32, [None, None], name="output_0")

        self.sequence_length1 = tf.placeholder(tf.int32, [None], name="seq_len_1")
        self.sequence_length0 = tf.placeholder(tf.int32, [None], name="seq_len_0")

        self.y1 = tf.placeholder(tf.float32, [None], name="label_1")
        self.y0 = tf.placeholder(tf.float32, [None], name="label_0")

        y1 = tf.expand_dims(self.y1, 1)
        y0 = tf.expand_dims(self.y0, 1)

        emb_W = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), self.embedding_size]),
                        trainable=False, name="Embed")
        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    shape=[len(self.vocab), self.embedding_size])
        self.embedding_init = emb_W.assign(self.embedding_placeholder)

        self.embed1 = tf.nn.embedding_lookup(self.embedding_placeholder, self.X1)
        self.embed0 = tf.nn.embedding_lookup(self.embedding_placeholder, self.X0)

        self.embed_dec1 = tf.nn.embedding_lookup(self.embedding_placeholder, self.decode_X1)
        self.embed_dec0 = tf.nn.embedding_lookup(self.embedding_placeholder, self.decode_X0)

        self.y_size = self.hidden_size - self.z_size

        self.y1_real = tf.layers.dense(y1, self.y_size)
        self.y0_real = tf.layers.dense(y0, self.y_size)

        # don't know if it causes any problems
        self.y1_fake = self.y0_real
        self.y0_fake = self.y1_real

        batch_size = tf.shape(self.X0)[0]

        init_y1 = tf.concat([self.y1_real, tf.zeros([batch_size, self.z_size])], 1)
        init_y0 = tf.concat([self.y0_real, tf.zeros([batch_size, self.z_size])], 1)

        # encoder
        enc_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=int(self.hidden_size / 2), state_is_tuple=False)
        cell_drop = tf.nn.rnn_cell.DropoutWrapper(enc_cell, input_keep_prob=self.keep_ratio)
        self.enc_network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers, state_is_tuple=False)

        # generator
        gen_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=int(self.hidden_size / 2), state_is_tuple=False, reuse=False)
        gen_cell_drop = tf.nn.rnn_cell.DropoutWrapper(gen_cell, input_keep_prob=self.keep_ratio)
        self.gen_network = tf.nn.rnn_cell.MultiRNNCell([gen_cell_drop] * self.num_layers, state_is_tuple=False)

        # making the encoding with the labels as the initialize state
        _, self.Z1 = tf.nn.dynamic_rnn(self.enc_network, self.embed1, dtype=tf.float32,
                                       initial_state=init_y1,
                                       sequence_length=self.sequence_length1, scope="encoder1")
        _, self.Z0 = tf.nn.dynamic_rnn(self.enc_network, self.embed0, dtype=tf.float32,
                                       initial_state=init_y0,
                                       sequence_length=self.sequence_length0, scope="encoder0")

        self.Z0_logits = tf.layers.dense(self.Z0, 2)
        self.Z1_logits = tf.layers.dense(self.Z1, 2)

        self.Z0_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.y0, tf.int32),
                                                                  logits=self.Z0_logits)
        self.Z1_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.y1, tf.int32),
                                                                  logits=self.Z1_logits)
        Z0_loss = tf.reduce_mean(self.Z0_xentropy)
        Z1_loss = tf.reduce_mean(self.Z1_xentropy)
        self.enc_loss = Z0_loss + Z1_loss

        real_Z0 = tf.concat([self.y0_real, self.Z0[:, : -self.y_size]], 1)
        real_Z1 = tf.concat([self.y1_real, self.Z1[:, : -self.y_size]], 1)
        fake_Z0 = tf.concat([self.y0_fake, self.Z0[:, : -self.y_size]], 1)
        fake_Z1 = tf.concat([self.y1_fake, self.Z1[:, : -self.y_size]], 1)

        self.gen1_outputs, self.gen1_state = tf.nn.dynamic_rnn(self.gen_network,
                                                               self.embed_dec1, dtype=tf.float32,
                                                               sequence_length=self.sequence_length1,
                                                               initial_state=real_Z1, scope="generator1")

        self.gen0_outputs, self.gen0_state = tf.nn.dynamic_rnn(self.gen_network,
                                                               self.embed_dec0, dtype=tf.float32,
                                                               sequence_length=self.sequence_length0,
                                                               initial_state=real_Z0, scope="generator0")

        self.logits1 = tf.layers.dense(self.gen1_outputs, len(self.vocab))
        self.logits0 = tf.layers.dense(self.gen0_outputs, len(self.vocab))

        self.gen1 = tf.argmax(self.logits1, 2)
        self.gen0 = tf.argmax(self.logits0, 2)

        self.xentropy_rec0 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.output_X0, logits=self.logits0)
        self.xentropy_rec1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.output_X1, logits=self.logits1)

        self.rec_loss = tf.reduce_mean(self.xentropy_rec0) + tf.reduce_mean(self.xentropy_rec1)
        self.rec_step = tf.train.AdamOptimizer(learning_rate=self.rec_learning_rate).minimize(self.rec_loss)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_placeholder,
            tf.fill([batch_size], self.vocab.index("<go>")), self.vocab.index("<eos>"))

        projection_layer = tf.layers.Dense(len(self.vocab), use_bias=False)

        generator0 = tf.contrib.seq2seq.BasicDecoder(
            gen_cell, helper, fake_Z1, output_layer=projection_layer)
        generator1 = tf.contrib.seq2seq.BasicDecoder(
            gen_cell, helper, fake_Z0, output_layer=projection_layer)

        self.generated0, _, seq_len0 = tf.contrib.seq2seq.dynamic_decode(
            generator0, maximum_iterations=20)
        self.generated1, _, seq_len1 = tf.contrib.seq2seq.dynamic_decode(
            generator1, maximum_iterations=20)

        disc1_loss, gen1_loss = discriminator(self.logits1, self.generated1.rnn_output, self.filter_sizes, self.num_filters, self.keep_ratio, scope="disc1")
        disc0_loss, gen0_loss = discriminator(self.logits0, self.generated0.rnn_output, self.filter_sizes, self.num_filters, self.keep_ratio, scope="disc0")

        self.disc_loss = disc0_loss + disc1_loss
        self.gen_loss = gen0_loss + gen1_loss

        self.enc_step = tf.train.AdamOptimizer(learning_rate=self.enc_learning_rate).minimize(-self.enc_loss)
        self.gen_step = tf.train.AdamOptimizer(learning_rate=self.gen_learning_rate).minimize(self.gen_loss) #, var_list=self.gen_vars)  # G Train step
        self.disc_step = tf.train.GradientDescentOptimizer(learning_rate=self.disc_learning_rate).minimize(self.disc_loss) #, var_list=self.disc_vars)  # D Train step

    def adversarial_train(self, batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            losses = {"discriminator": list(),
                      "generator": list(),
                      "recogniztion": list(),
                      "autoencoder": list()}
            while True:
                disc_loss, gen_loss, rec_loss, enc_loss = 0, 0, 0, 0
                #_ = self.sess.run(self.embedding_init,
                #                  feed_dict = {self.embedding_placeholder: self.embeddings})
                for (target, domain) in batches:
                    feed_dict = {
                        self.X1: [t["enc_input"] for t in target],
                        self.X0: [d["enc_input"] for d in domain],
                        self.decode_X1: [t["dec_input"] for t in target],
                        self.decode_X0: [d["dec_input"] for d in domain],
                        self.sequence_length1: [t["length"] for t in target],
                        self.sequence_length0: [d["length"] for d in domain],
                        self.output_X1: [t["enc_output"] for t in target],
                        self.output_X0: [d["enc_output"] for d in domain],
                        self.y1: [1 for t in target],
                        self.y0: [0 for d in domain],
                        self.keep_prob: self.keep_ratio,
                        self.embedding_placeholder: self.embeddings
                    }

                    if epoch < 50:
                        _, _, rec_l, enc_l = self.sess.run(
                            [self.rec_step, self.enc_step,
                             self.rec_loss, self.enc_loss],
                            feed_dict=feed_dict)
                        rec_loss += rec_l
                        enc_loss += enc_l
                    else:
                        _, _, _, _, gen_l, disc_l, rec_l, enc_l = self.sess.run(
                            [self.gen_step, self.disc_step, self.rec_step, self.enc_step,
                             self.gen_loss, self.disc_loss, self.rec_loss, self.enc_loss],
                            feed_dict=feed_dict)
                        gen_loss += gen_l
                        disc_loss += disc_l
                        rec_loss += rec_l
                        enc_loss += enc_l

                print("Iterations: %d\n Recognition loss: %.4f"
                      "\n Autoencoder loss: %.4f\n Generator loss: %.4f"
                      "\n Discriminator loss: %.4f" %
                      (epoch, rec_loss / len(batches), enc_loss / len(batches),
                       gen_loss / len(batches), disc_loss / len(batches)))
                losses["discriminator"].append(disc_loss / len(batches))
                losses["autoencoder"].append(enc_loss / len(batches))
                losses["generator"].append(gen_loss / len(batches))
                losses["recogniztion"].append(rec_loss / len(batches))
                epoch += 1
                if epoch == self.epochs:
                    # It's the output sentence of the auto-encoder
                    z = self.gen0.eval(feed_dict=feed_dict)
                    # The output sentence after changing the style
                    y = self.generated1.sample_id.eval(feed_dict=feed_dict)

                    for s in range(len(list(z))):
                        print("sent:", " ".join([self.vocab[int(x)] for x in list(feed_dict[self.X0][s])]))
                        print("encod:"," ".join([self.vocab[int(x)] for x in list(list(z)[s])]))
                        print("trans:"," ".join([self.vocab[int(x)] for x in list(list(y)[s])]))

                    print('\n')
                    z = self.gen1.eval(feed_dict=feed_dict)
                    y = self.generated0.sample_id.eval(feed_dict=feed_dict)
                    for s in range(len(list(z))):
                        print("sent:", " ".join([self.vocab[int(x)] for x in list(feed_dict[self.X1][s])]))
                        print("encod:", " ".join([self.vocab[int(x)] for x in list(list(z)[s])]))
                        print("trans:", " ".join([self.vocab[int(x)] for x in list(list(y)[s])]))
                    saver.save(self.sess, "saved_model/model")
                    pd.DataFrame.from_dict(losses).to_csv("losses.csv")
                    break


