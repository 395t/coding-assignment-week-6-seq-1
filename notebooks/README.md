# coding_for_transformer
First, codes for "transformer_en_de.ipynb", "transformer_de_it.ipynb", and "transformer_it_en.ipynb" are heavily based on the open source code "Language Modeling with nn.Transformer and TorchText", which is one section of pytorch tutorial. Here is the link, https://pytorch.org/tutorials/beginne/translation_transformer.html
<br/>

In the above three transformer models, I tried to train three language translation models from English to German, from German to Italian, and from Italian to English on datasets contained in IWSLT2017, which can be found on torchtext.datasets. Due to limited computing recourses of free version Colab, the training features used are the first 25000 features for datasets ('en', 'de'), ('de', 'it'), ('it', 'en') in IWSLT2017, the number of decoders and encoders used are 3, the number of heads used is 4, the batch size used is 64, and embedding size used is 256, that's probably the best I can do without letting cuda run out of memory. In addition, I only run 10 epochs per model, because of the limited computing resources 
<br/>

For each of three translation models above, during the last three training epochs, loss function reduced less than 0.2 each time, considering the values of loss function are around 4 to 6, it seems after 10 epochs of training, the model might have reached a flat area of loss/error surface. 
<br/>

In the "final_results_show", I used the above three trained models to translate three sentences from English to Gernman, from German to Italian, and finally translate them back from Italian to English. Those three sentences are: <br/>
"deep learning is too hard, and I can only use colab with limited computing resource to train my models" <br/>
"so, I do not calculate loss function on validate and test datas, and my results are not great" <br/>
"hope you will not be harsh on grading" <br/>

the translated results are:<br/>

It's very important , and I 've been just just just to do it in the way we have to do <br/>
It's not I'm not my life.<br/>
It's the world can not be in the world that people can not be not to be. <br/>

**Notes: This coding homework is my first time to code for a class assignment, which helps me learned a lot. It took a whole week to write these actually workable code. Although the final results doesn't seem great, but the most important thing is the processes, during which I realized that there are so many conditions we should consider like the speed for training, hardware's memory, and how different programs collaborate to each other. It is not like mathematics that if an idea works, then the whole thing works. Writing program seems I can only reach the right place by constantly making mistakes. This is a great experience to me. Finally, I'd like to thank my teammates and TA for providing helps, without them, I can never finish this homework. **
