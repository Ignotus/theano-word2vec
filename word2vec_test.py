#!/usr/bin/env python2
import sys
import cPickle as pickle
from word2vec import *


if __name__ == '__main__':
    corpus = Corpus('data/', vocab_size=8000, corpus_file='corpus')

    vector_size = 150
    context_window_size = 5
    # https://groups.google.com/forum/#!msg/word2vec-toolkit/O9YUT0p5rCw/LIjsunElNm4J
    epochs = 10
    learning_rate = 0.05
    batch_size = 2

    skip_gram = (len(sys.argv) == 1 or sys.argv[1] == '--skip_gram')

    if skip_gram:
        print('SkipGram')
        trainer = SkipGram(vector_size, corpus)
        prefix = 'skip_gram'
    else:
        print('CBOW')
        trainer = CBOW(vector_size, corpus)
        prefix = 'cbow'
    loss, loss_changes = trainer.train(context_window_size, learning_rate, epochs, batch_size)

    trainer.save('%s.npy' % prefix)
    with open('%s_losses.npy' % prefix, 'w') as f:
        pickle.dump(loss_changes, f)
