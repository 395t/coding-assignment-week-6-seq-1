# Neural Machine Translation by Jointly Learning to Align and Translate

Coder: Elias Lampietti, ejl2425

This code was adapted from Sean Robertson's seq2seq translation network linked [here](https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py)

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

## Seq2seq Translation Training



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
