import numpy as np
import tensorflow as tf
from nn import *

class GAN():
    def __init__(self, params, domain_df, target_df, vocab):
        self.vocab = vocab
        self.domain = domain_df
        self.target = target_df
        print("Domain sample shape:", len(self.domain))
        #print("Domain is high on", params["generate_domain"])
        print("Target sample shape:", len(self.target))
        #print("Target is high on", params["generate_target"])

        self.params = params
        for key in params:
            setattr(self, key, params[key])
        self.adversarial_build()
        self.target, self.domain = balance_data(self.target, self.domain)
        batches = get_batches(self.target, self.domain, self.batch_size, vocab.index("<pad>"), vocab.index("<eos>"), vocab.index("<go>"))
        self.adversarial_train(batches)


    def adversarial_build(self):
        tf.reset_default_graph()
        self.keep_prob = tf.placeholder(tf.float32)

        self.X1 = tf.placeholder(tf.int32, [None, None])
        self.X0 = tf.placeholder(tf.int32, [None, None])

        self.maximum_length = 2 * self.X1.shape[1]

        self.decode_X1 = tf.placeholder(tf.int32, [None, None])
        self.decode_X0 = tf.placeholder(tf.int32, [None, None])

        self.output_X1 = tf.placeholder(tf.int32, [None, None])
        self.output_X0 = tf.placeholder(tf.int32, [None, None])

        self.sequence_length1 = tf.placeholder(tf.int32, [None])
        self.sequence_length0 = tf.placeholder(tf.int32, [None])

        self.y1 = tf.placeholder(tf.float32, [None])
        self.y0 = tf.placeholder(tf.float32, [None])

        y1 = tf.expand_dims(self.y1, 1)
        y0 = tf.expand_dims(self.y0, 1)

        self.embedding_placeholder = build_embedding(len(self.vocab), self.embedding_size, self.pretrain)

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

        self.loss_rec = tf.reduce_mean(self.xentropy_rec0) + tf.reduce_mean(self.xentropy_rec1)
        self.rec_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_rec)

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

        self.gen_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.gen_loss) #, var_list=self.gen_vars)  # G Train step
        self.disc_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.disc_loss) #, var_list=self.disc_vars)  # D Train step

    def adversarial_train(self, batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            while True:
                disc_loss = 0
                gen_loss = 0
                rec_loss = 0
                batch = 0
                print(epoch)
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
                        self.keep_prob: self.keep_ratio
                    }

                    # It's the output sentence of the auto-encoder
                    z = self.gen0.eval(feed_dict=feed_dict)
                    # The output sentence after changing the style
                    y = self.generated1.sample_id.eval(feed_dict=feed_dict)

                    for s in range(len(list(z))):
                        print("sent:", " ".join([self.vocab[int(x)] for x in list(feed_dict[self.X0][s])]))
                        print("encod:"," ".join([self.vocab[int(x)] for x in list(list(z)[s])]))
                        print("trans:"," ".join([self.vocab[int(x)] for x in list(list(y)[s])]))

                    print("\n")
                    z = self.gen1.eval(feed_dict=feed_dict)
                    y = self.generated0.sample_id.eval(feed_dict=feed_dict)
                    for s in range(len(list(z))):
                        print("sent:", " ".join([self.vocab[int(x)] for x in list(feed_dict[self.X1][s])]))
                        print("encod:", " ".join([self.vocab[int(x)] for x in list(list(z)[s])]))
                        print("trans:", " ".join([self.vocab[int(x)] for x in list(list(y)[s])]))

                    if epoch < 1:
                        _, gen_l, disc_l, rec_l = self.sess.run([self.rec_step, self.gen_loss, self.disc_loss, self.loss_rec], feed_dict=feed_dict)
                    else:
                        _, _, _, gen_l, disc_l, rec_l = self.sess.run(
                            [self.gen_step, self.disc_step, self.rec_step, self.gen_loss, self.disc_loss, self.loss_rec], feed_dict=feed_dict)
                    #gen_loss += gen_l
                    #disc_loss += disc_l
                    rec_loss += rec_l
                    batch += 1
                print("Iterations: %d\t Recognition loss: %.4f\t Discriminator loss: %.4f\t Generator loss: %.4f" % (epoch, rec_loss / len(batches),  disc_loss / len(batches), gen_loss / len(batches)))

                epoch += 1
                if epoch == 100:
                    break


