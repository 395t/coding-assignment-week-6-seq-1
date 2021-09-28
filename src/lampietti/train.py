import math
import torch.nn as nn
import torch.nn.functional as F
from model import Teacher, Seq2Seq
from eval import eval_model
from torch import optim
from hyper_params import device, subset_size, batch_size, lr

def train_model(encoder, decoder, training_pairs, validation_pairs, epochs):
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    
    decoding_helper = Teacher(teacher_forcing_ratio=0.5)

    epoch_losses = []
    valid_losses = []

    optimizer = optim.SGD(seq2seq.parameters(), lr=lr)

    # compute total number of iterations
    n_iters = math.ceil(subset_size / batch_size)

    for epoch in range(epochs):
        seq2seq.train()
        total_loss = 0
        for iter in range(1, n_iters + 1):
            input_batch, target_batch = training_pairs.get_batch(batch_size, iter-1)
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            decoding_helper.load_targets(target_batch)
            outputs, masks = seq2seq(input_batch, decoding_helper)

            loss = F.cross_entropy(outputs.view(-1, outputs.size(2)),
                        target_batch[1:].reshape(-1), ignore_index=1)

            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq.parameters(), 10.0, norm_type=2)  # prevent exploding grads

            optimizer.step()

            total_loss += loss.item()
    
        print("Epoch {} finished with avg loss {}".format(epoch, total_loss / n_iters))
        # store average loss for this epoch
        epoch_losses.append(total_loss / n_iters)

        # test on validation set
        valid_loss = eval_model(seq2seq, validation_pairs)
        valid_losses.append(valid_loss)

    return seq2seq, epoch_losses, valid_losses, optimizer