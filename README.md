# lapse-elmo
Distributed training of an ELMo NLP model based on [allennlp-elmo](https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py), [bilm-tf](https://github.com/allenai/bilm-tf) and [pytorch-fast-elmo](https://github.com/huntzhan/pytorch-fast-elmo).

In contrast to previous ELMo implementations, the model uses a direct embedding layer instead of a character CNN for the initial word embedding. This makes the majority of model parameters sparse and allows for efficient distributed training.

## Get started
### Dependencies
The model is implemented in the [PyTorch](https://pytorch.org/) framework and uses multiple ELMo related modules of the [AllenNLP](https://allennlp.org/) library. Install both via the following commands:
```bash
pip3 install torch torchvision torchaudio
pip3 install numpy spacy  # allennlp requirements
pip3 install allennlp
```

The distributed training relies on the parameter server [Lapse](https://github.com/alexrenz/lapse-ps). Use the following commands to fetch required dependencies, build the parameter server and install the python bindings:
```bash
sudo apt-get update && sudo apt-get install -y build-essential git libboost-all-dev
git clone https://github.com/alexrenz/lapse-ps.git
cd lapse-ps
make ps KEY_TYPE=int64_t CXX11_ABI=$(python bindings/lookup_torch_abi.py) DEPS_PATH=$(pwd)/deps_bindings
cd bindings
python3 setup.py install --user
```

### Setup
Simply clone the git repository:
```bash
git clone https://github.com/robertgericke/lapse-elmo.git
```
### Training data
Fetch the training data from the [1 Billion Word Benchmark](http://www.statmt.org/lm-benchmark/) and either download a precompiled [vocabulary file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt) or create one using the vocabulary utility:
```bash
# Download training data
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -xzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
# Download vocabulary file
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt
# Create vocabulary file using the vocab utility
python3 vocab.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/
```

Note: The vocabulary file encompasses tokens found in the training data each separated by a new line and sorted by frequency - highest to lowest, but it does not require any special tokens e.g. `<S>,</S>` which other implementations use to mark sentence boundaries.

### Running the training script
```bash
cd lapse-elmo
python3 run.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab-2016-09-10.txt
```

Note: See `python3 run.py --help` for further programm options.
