import tensorflow as tf
import numpy as np
from nltk import tokenize as nltk_token
import operator
import re


def generator(Z, hidden_size, vocab_len, label):
    with tf.variable_scope("GAN/Generator", reuse=tf.AUTO_REUSE):
        h1 = tf.layers.dense(Z, hidden_size, activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h1, vocab_len, activation=tf.tanh)
    return out


def discriminator(X_real, X_fake, filter_sizes, num_filters, keep_ratio, scope, label):
    cnn_real = cnn(X_real, filter_sizes, num_filters, keep_ratio, scope)
    cnn_fake = cnn(X_fake, filter_sizes, num_filters, keep_ratio, scope, reuse=True)

    class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(cnn_real) if label == 1 else tf.zeros_like(cnn_real)
        , logits=cnn_real))

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(cnn_real), logits=cnn_real)) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                 labels=tf.zeros_like(cnn_fake), logits=cnn_fake))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(cnn_real), logits=cnn_fake))
    return disc_loss, gen_loss, class_loss

def cnn(input, filter_sizes, num_filters, keep_ratio, scope, padding="VALID", reuse=False):
    dim = input.get_shape().as_list()[-1]
    input = tf.expand_dims(input, -1)

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        pooled_outputs = list()
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                b = tf.get_variable('b', shape=[num_filters])
                W = tf.get_variable('W', shape=[size, dim, 1, num_filters])

                conv = tf.nn.conv2d(input, W,
                    strides=[1, 1, 1, 1], padding=padding)
                relu = tf.nn.relu(tf.nn.bias_add(conv, b))

                pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        drop = tf.nn.dropout(h_pool_flat, keep_ratio)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [num_filters * len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(drop, W) + b, [-1])

    return logits

def balance_data(target, domain):
    if len(target) > len(domain):
        # domain = domain.sample(len(target))
        #x = domain
        #max_len = len(target)
        print("Extending domain")
        domain = extend(domain, len(target))
    elif len(target) < len(domain):
        # target = target.sample(len(domain))
        #x = target
        #max_len = len(domain)
        print("Extending target")
        target = extend(target, len(domain))
    return target, domain

def extend(data, length):
    new = list()
    while len(new) < length:
        new.extend(data[:min(len(data), length - len(new))])
    print("Extending to ", length)
    print("New length is", len(new))
    return new

def get_batches(target, domain, batch_size, pad_idx, eos_idx, go_idx):
    batches = []
    for idx in range(len(target) // batch_size + 1):
        if idx * batch_size !=  len(target):
            target_batch = target[idx * batch_size: min((idx + 1) * batch_size, len(target))]
            target_info = batch_to_info(target_batch, pad_idx, eos_idx, go_idx, target=True)

            domain_batch = domain[idx * batch_size: min((idx + 1) * batch_size, len(domain))]
            domain_info = batch_to_info(domain_batch, pad_idx, eos_idx, go_idx, target=False)

        batches.append((target_info, domain_info))
    return batches

def batch_to_info(batch, pad_idx, eos_idx, go_idx, target=False):
    max_len = max(len(sent) for sent in batch)
    batch_info = list()
    for sent in batch:
        padding = [pad_idx] * (max_len - len(sent))
        sentence = {
            "enc_input": padding + sent[::-1],
            "enc_output": sent + [eos_idx] + padding,
            "dec_input": [go_idx] + sent + padding,
            "length": len(sent),
            "labels": [1 for i in range(max_len)] if target else [0 for i in range(max_len)]
        }
        batch_info.append(sentence)
    return batch_info

def tokenize_data(corpus, col):
    #sent_tokenizer = toks[self.params["tokenize"]]
    for idx, row in corpus.iterrows():
        corpus.at[idx, col] = nltk_token.WordPunctTokenizer().tokenize(clean(row[col]))
    return corpus

def clean(sent):
    return re.sub(r'[^a-zA-Z ]', r'', sent.lower())


def remove_empty(corpus, col):
    drop = list()
    for i, row in corpus.iterrows():
        if row[col] == "" or len(row[col]) < 4 or len(row[col]) > 100:
            drop.append(i)
    return corpus.dropna().drop(drop)

def learn_vocab(corpus, vocab_size):
    print("Learning vocabulary of size %d" % (vocab_size))
    tokens = dict()
    for sent in corpus:
        for token in sent:
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    words, counts = zip(*sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
    return list(words[:vocab_size]) + ["<unk>", "<pad>", "<go>", "<eos>"]

def tokens_to_ids(corpus, vocab):
    print("Converting corpus of size %d to word indices based on learned vocabulary" % len(corpus))
    if vocab is None:
        raise ValueError("learn_vocab before converting tokens")

    mapping = {word: idx for idx, word in enumerate(vocab)}
    unk_idx = vocab.index("<unk>")
    for i in range(len(corpus)):
        row = corpus[i]
        for j in range(len(row)):
            try:
                corpus[i][j] = mapping[corpus[i][j]]
            except:
                corpus[i][j] = unk_idx
    return corpus

def load_embedding(vocabulary, file_path, embedding_size):
    embeddings = np.random.randn(len(vocabulary), embedding_size)
    found = 0
    with open(file_path, "r") as f:
        for line in f:
            split = line.split()
            idx = len(split) - embedding_size
            vocab = "".join(split[:idx])
            if vocab in vocabulary:
                embeddings[vocabulary.index(vocab)] = np.array(split[idx:], dtype=np.float32)
                found += 1
    print("Found {}/{} of vocab in word embeddings".
          format(found, len(vocabulary)))
    return embeddings
