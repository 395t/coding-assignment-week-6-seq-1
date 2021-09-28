# Neural Machine Translation by Jointly Learning to Align and Translate

Coder: Elias Lampietti, ejl2425

Implementation of Bahdanau, Cho, and Bengio's [paper](https://arxiv.org/pdf/1409.0473v7.pdf) from 2014

This code was adapted from Austin Jacobson's neural machine translation network linked [here](https://github.com/A-Jacobson/minimal-nmt)

## Requirements
* Python 3.9.5
* PyTorch 1.9.1
* Torchtext 0.10.1
* Spacy 3.1.3
* NLTK 3.6.2
* Scipy 1.6.3
* Matplotlib 3.4.2
* tqdm 4.60.0

## Data Prerequisites
```bash
python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm
python3 -m spacy download it_core_news_sm
```

## RNNsearch Model

The model builds off of the standard encoder-decoder model for neural machine translation. The paper points out a bottleneck in these models by encoding the source sentence into a fixed length vector. To achieve better performance, especially with longer sentences, the input sentence is instead encoded into a sequence of context vectors that the decoder then adaptively picks out a subset from. 

The model consists of a bidirectional RNN as the encoder for the source sentence, and a decoder that imitates searching through this source sentence while it is decoding encoded version into a translation.

## Training

The model was trained on 3 different datasets from IWSLT2017. These datasets were English (en) to German (de), German to Italian (it), and Italian to English.
A subset of 25,000 samples was taken from each dataset for training for 10 epochs.
The models were then evaluated during training on the validation sets (~1,000 samples).
The average loss (NLL) was calculated after each epoch of the training data as well as once through the validation data.

## Training Results

The following results were obtained on 25,000 training samples and ~1,000 validation samples.
For all models, the rate at which the loss decreases slows down at around the fourth epoch. Also the validation and training losses have similar trajectories.
Given stronger computing resources and more time, it is likely much lower losses could be achieved with more samples and epochs.

![en_de](https://user-images.githubusercontent.com/7085644/135036193-e2af7a2f-3e2e-4d58-bd78-4151a86eeb0b.png)

![de_it](https://user-images.githubusercontent.com/7085644/135036204-6e3f2224-71fb-45d4-8d6a-092d77dd5154.png)

![it_en](https://user-images.githubusercontent.com/7085644/135036218-73b9181e-6155-442b-85d2-85e3edf1ff4d.png)

## Testing Results



## Data Sources

For the IWSLT2017 datasets, refer to the [PyTorch tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html) for preprocessing code used in this repository. The IWSLT2017 dataset is from the paper [WIT3: Web Inventory of Transcribed and Translated Talks](https://aclanthology.org/2012.eamt-1.60.pdf). You can cite their work with:
```
@inproceedings{cettoloEtAl:EAMT2012,
    Address = {Trento, Italy},
    Author = {Mauro Cettolo and Christian Girardi and Marcello Federico},
    Booktitle = {Proceedings of the 16$^{th}$ Conference of the European Association for Machine Translation (EAMT)},
    Date = {28-30},
    Month = {May},
    Pages = {261--268},
    Title = {WIT$^3$: Web Inventory of Transcribed and Translated Talks},
    Year = {2012},
}
```
