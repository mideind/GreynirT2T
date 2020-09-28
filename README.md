# GreynirT2T
Machine Translation between Icelandic and English using [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).
Most of the provided scripts assume they are running inside a docker container, although they should run outside containers as well.
Many of the paths are hard-coded and need to be adapted (see e.g. greynirt2t/translate_enis.py)

## Data ##
Bilingual parallel data and monolingual Icelandic data can be downloaded from [CLARIN](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/16 "CLARIN").
Note that a license must be accepted, and the OpenSubtitles2018 subcorpus must be downloaded and cleaned (by a provided script).

### Data preparation ###
Use the cleaning and preprocessing script filters.py to optionally clean data before training (see script).

### Vocabulary ###
In order to run the pre-trained models, the vocabulary used at training time must be used. 

## Training ##
Before training can begin, the training data must be binarized and a vocabulary must be generated (if one does not already exist).

Note that batch size must be tuned according to your GPU by trial and error since it depends on available VRAM, model size, and maximum sequence size.
If a larger batch size is wanted than can fit on your GPU, then larger batch sizes can be simulated (with multistep_adam, see scripts/env.sh).
The training can be found at greynirt2t/scripts/train.sh.

## Inference / translation ##
To view model predictions, see interactive_decode.sh or translate_file.sh (or the T2T repository for documentation).
