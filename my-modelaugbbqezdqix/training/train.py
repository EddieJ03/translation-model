import math
import sys
import time
import os

import joblib

# add parent directory to path
# add to original cookiecutter if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from model.nmt_model import NMT
import numpy as np
from .utils import read_corpus, batch_iter
from .vocab import Vocab

import torch
import torch.nn.utils

import mlflow
import mlflow.pytorch

# List of required environment variables
REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "AZURE_STORAGE_CONNECTION_STRING"]

# Check if all required variables are set
missing_vars = [var for var in REQUIRED_ENV_VARS if var not in os.environ]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# If all required variables are set, proceed
print("All required environment variables are set.")

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

def train(train_batch_size=32, clip_grad=5.0, log_every=10, dropout=0.3, device='cpu', lr=5e-5, max_epoch=30, max_patience=5, uniform_init=0.1, lr_decay=0.5, max_decoding_time_step=70, max_num_trial=5):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Move two directories up
    data_dir = os.path.join(current_dir, "data")

    src_vocab = os.path.join(data_dir, "train.zh")
    tgt_vocab = os.path.join(data_dir, "train.en")
    
    train_data_src, spm_src = read_corpus(src_vocab, os.path.join(data_dir,'{}.model'.format('src')), vocab_size=21000)  
    train_data_tgt, _ = read_corpus(tgt_vocab, os.path.join(data_dir,'{}.model'.format('tgt')), vocab_size=8000)

    train_data = list(zip(train_data_src, train_data_tgt))

    vocab = Vocab.load(os.path.join(data_dir, "vocab.json"))

    model = NMT(embed_size=1024,
                hidden_size=768,
                dropout_rate=dropout,
                vocab=vocab)
    
    model.train()

    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_iter = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = 0
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    with mlflow.start_run():
        mlflow.log_param("epochs", max_epoch)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", lr)
        
        while not (epoch == max_epoch):
            epoch += 1

            for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
                train_iter += 1

                optimizer.zero_grad()

                batch_size = len(src_sents)

                example_losses = -model(src_sents, tgt_sents) # (batch_size,)
                batch_loss = example_losses.sum()
                loss = batch_loss / batch_size

                loss.backward()

                # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                optimizer.step()

                batch_losses_val = batch_loss.item()
                report_loss += batch_losses_val
                cum_loss += batch_losses_val

                tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
                report_tgt_words += tgt_words_num_to_predict
                cum_tgt_words += tgt_words_num_to_predict
                report_examples += batch_size
                cum_examples += batch_size

                if train_iter % log_every == 0:
                    mlflow.log_metric("avg_loss", report_loss / report_tgt_words, step=train_iter)
                    mlflow.log_metric("avg_ppl", math.exp(report_loss / report_tgt_words), step=train_iter)
                    mlflow.log_metric("cum_examples", cum_examples, step=train_iter)
                    mlflow.log_metric("speed_words_per_sec", report_tgt_words / (time.time() - train_time), step=train_iter)
                    mlflow.log_metric("elapsed_time_sec", time.time() - begin_time, step=train_iter)

                    train_time = time.time()
                    report_loss = report_tgt_words = report_examples = 0.
                    
                    if train_iter == 4000:
                        break

        mlflow.pytorch.log_model(model, "models")
        spm_src_path = "spm_src.pkl"
        joblib.dump(spm_src, spm_src_path)

        # Log the TfidfVectorizer as an artifact
        mlflow.log_artifact(spm_src_path)

def main():
    train(max_epoch=1)

if __name__ == '__main__':
    main()
