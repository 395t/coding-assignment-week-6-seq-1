# Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling

This directory contains the code for the work [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555).

## Requirements
* Python 3.6.13
* PyTorch 1.9.0
* Spacy 2.0.12

## Data Prerequisites
```bash
python3 -m spacy download en_core_web_sm
python3 -m spacy download en_core_news_sm
```

## Data Sources
For the music datasets (JSB, MuseData, Piano-midi, Nottingham), refer to the [TCN repository](https://github.com/locuslab/TCN/tree/master/TCN/poly_music). TCN, or Temporal Convolutional Networks, can be found in the work [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271). You can cite their work with:

```
@article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
}
```
