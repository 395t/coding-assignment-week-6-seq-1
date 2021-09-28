import torch.nn.functional as F
import math
from model import Greedy
from nltk.translate.bleu_score import sentence_bleu
from hyper_params import batch_size, device

def eval_model(model, test_pairs, bleu=False):
    # Get test results
    model.eval()
    
    # Use greedy decoding helper to choose highest score
    decoding_helper_greedy = Greedy()
    decoding_helper_greedy.set_maxlen(49)
    # compute total number of iterations
    n_iters = math.ceil(len(test_pairs) / batch_size)
    
    # Track loss and bleu score
    total_loss = 0
    bleu_scores = []
    bleu_score = None

    # evaluate on test set
    for iter in range(1, n_iters + 1):
        input_batch, target_batch = test_pairs.get_batch(batch_size, iter-1)

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        outputs, masks = model(input_batch, decoding_helper_greedy)

        loss = F.cross_entropy(outputs.view(-1, outputs.size(2)),
                           target_batch[1:].reshape(-1), ignore_index=1)

        total_loss += loss.item()

        if bleu:
            preds = outputs.topk(1)[1].squeeze(2)
            # source = test_pairs.convert_src_ids_to_text(input_batch[:, 0].tolist())
            # prediction = test_pairs.convert_tgt_ids_to_text(preds[:, 0].tolist())
            # target = test_pairs.convert_tgt_ids_to_text(target_batch[1:, 0].tolist())
            # print("source: ", source)
            # print("prediction: ", prediction)
            # print("target: ", target)

            for idx in range(0, preds[0].shape[0]):
                ref = target_batch[1:, idx].tolist()
                candidate = preds[:, idx].tolist()
                score = sentence_bleu([ref], candidate)
                bleu_scores.append(score)

    avg_test_loss = total_loss / n_iters
    print("\navg_test_loss {}\n".format(avg_test_loss))
    if bleu:
        bleu_score = sum(bleu_scores) / len(bleu_scores)
        print("bleu score: {}\n".format(bleu_score))
    return avg_test_loss, bleu_score