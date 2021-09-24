# Polyphonic Music Datasets

Source: https://github.com/locuslab/TCN/tree/master/TCN/poly_music/mdata

The original source of the data can be found at: http://www-ens.iro.umontreal.ca/~boulanni/icml2012

Each dataset contains sequences of notes to be played at each timestep, and instead of notes, they are preprocessed one-hot vector of notes. There are 88 possible notes in total, which corresponds to MIDI note numbers between 21 and 108 inclusive. Refer to the notebook `music_data_check.ipynb` in the notebooks directory for more information.
