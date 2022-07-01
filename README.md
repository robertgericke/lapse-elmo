# lapse-elmo
Distributed training of an ELMo NLP model based on [allennlp-elmo](https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py), [bilm-tf](https://github.com/allenai/bilm-tf) and [pytorch-fast-elmo](https://github.com/huntzhan/pytorch-fast-elmo).

In contrast to previous ELMo implementations, the model uses a direct embedding layer instead of a character CNN for the initial word embedding. This makes the majority of model parameters sparse and allows for efficient distributed training.

## Get started
### Dependencies
The model is implemented in the [PyTorch](https://pytorch.org/) framework and uses multiple ELMo related modules of the [AllenNLP](https://allennlp.org/) library. Install both via the following command:
```bash
pip3 install allennlp torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

The distributed training relies on the parameter server [Lapse](https://github.com/alexrenz/lapse-ps). Use the following commands to fetch required dependencies, build the parameter server and install the python bindings:
```bash
sudo apt-get update && sudo apt-get install -y build-essential git libboost-all-dev
git clone https://github.com/alexrenz/lapse-ps.git
cd lapse-ps
make ps KEY_TYPE=int64_t CXX11_ABI=$(python bindings/lookup_torch_abi.py) DEPS_PATH=$(pwd)/deps_bindings
cd bindings
python setup.py install --user
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
python vocab.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/
```

Note: The vocabulary file encompasses tokens found in the training data each separated by a new line and sorted by frequency - highest to lowest, but it does not require any special tokens e.g. `<S>,</S>` which other implementations use to mark sentence boundaries.

### Running the training script
```bash
cd lapse-elmo
python run.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab-2016-09-10.txt
```
#### Local training options
Use the options `--nodes` and `--workers_per_node` to set the number of server processes and worker threads per process.
```bash
python run.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab-2016-09-10.txt --nodes 2 --workers_per_node 1
```
The example above starts two nodes with one worker each.

Note: since world_size is not provided the script automatically asumes local training and implicitly starts a scheduler.

#### CUDA options
The script automatically makes use of available CUDA devices. The this can be disabled by using `--no_cuda`.
By default workers are assigned round robin to CUDA devices. Use `--device_ids` to provide an alternative assignment (one device ID for each worker thread).
```bash
python run.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab-2016-09-10.txt --nodes 2 --workers_per_node 1 --device_ids 2 3
```

#### Distributed training
Begin distributed training by starting the parameter server scheduler explicitly. For that specify the address and port of the scheduler as well as the world size which is the total number of server nodes.
```bash
python run.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab-2016-09-10.txt --role scheduler --nodes 0 --root_uri "127.0.0.1" --root_port "9091" --world_size 2
```
After that start the specified number of nodes. As before, specify the address and port of the scheduler and the world size.
```bash
python run.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab-2016-09-10.txt --nodes 1 --root_uri "127.0.0.1" --root_port "9091" --world_size 2
```

#### Running with tracker scrips
The Script may also be strated using the [tracker scrips](https://github.com/alexrenz/lapse-ps/tree/main/tracker) of Lapse. For that use the `--tracker` option.
```bash
python ../lapse/tracker/dmlc_ssh.py -s 2 -H ../hosts.txt python run.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab-2016-09-10.txt --tracker
```


Note: See `python run.py --help` for further program options.
