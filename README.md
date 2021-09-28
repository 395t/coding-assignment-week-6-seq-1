# coding-template

## Summary

The summary can contain but is not limited to:

- Code structure.

- Commands to reproduce your experiments.

- Write-up of your findings and conclusions.

- Ipython notebooks can be organized in `notebooks`.

## Reference

Any code that you borrow or other reference should be properly cited.

For the sequence to sequence paper by Sutskever et al, the code from the below 2 sources were used:

https://pytorch.org/tutorials/beginner/translation_transformer.html

https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb


For the translation task we used the dataset IWSLT 2017 in torchtext and Multi30k.

https://pytorch.org/text/stable/datasets.html#iwslt2017

https://pytorch.org/text/stable/datasets.html#multi30k

For the Music task, we used the dataset


For the translation task, due to memory and hardware constraints, we use a subset of the IWSLT dataset. We use the first 25000 examples of the training, first 5000 of the validation and 5000 of the test dataset.

## Sequence to Sequence Learning using deep LSTM encoder decoder

English to German task

![LSTM training losses for English to German](./images/seq2seq_losses_epochs20_en_de.png)

German to Italian task

![LSTM training losses for German to Italian](./images/seq2seq_losses_epochs20_de_it.png)

Italian to English task

![LSTM training losses for Italian to English](./images/seq2seq_losses_epochs20_it_en.png)


The LSTMs overfit to the first 25000 examples causing it to do worse in the validation and test dataset. When running on a smaller dataset such as WMT'14, we did not see this issue.

English to German task on WMT'14

![LSTM training losses for English to German](./images/seq2seq_losses_epochs20_en_de_wmt14.png)


## Gated Recurrent Neural Networks on Sequence Modeling

German to Italian task

![Transformer training losses for English to German](./images/en_de_transformer.png)

English to German task

![Transformer training losses for German to Italian](./images/de_it_transformer.png)

Italian to English task

![Transformer training losses for Italian to English](./images/it_en_transformer.png)


## Neural Machine translation

English to German task

![training losses for English to German](./images/en_de.png)

German to Italian task

![training losses for German to Italian](./images/de_it.png)

Italian to English task

![training losses for Italian to English](./images/it_en.png)
