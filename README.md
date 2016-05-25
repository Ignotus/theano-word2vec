# word2vec-theano

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)

This is a close-to-vanilla implementation of the SkipGram and CBOW models (aka Word2Vec) described in the paper [Distributed Representations of Words and Phrases
and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean, following derivations described in [word2vec Parameter Learning Explained](word2vec Parameter Learning Explained) by Xin Rong.

## Deviations

* Support of batch training. Padding by UNK tokens at the beginning and at the end of sentences has made it possible.
* AdaGrad gradient update.
* The corpus is separated into two parts. 10% is used for validation.

## Dependencies

* theano
* lasagne
* numpy
* nltk

## Testing

```bash
python word2vec.py
```

## License

Copyright (c) 2016 Minh Ngo

The source code is distributed under the [MIT license](LICENSE).
