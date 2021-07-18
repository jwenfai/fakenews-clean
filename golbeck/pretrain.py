import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, AdamW
from nltk import sent_tokenize

# import getpass
# user = getpass.getuser()
# if user == 'Low':
#     import os
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# else:
#     import os


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MaskedLMTokenizer:
    # The training data generator chooses 15% of the token positions at random for prediction.
    # If the i-th token is chosen, we replace the i-th token with (1) the [MASK] token 80% of the time
    # (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time.
    def __init__(self, tokenizer_class, pretrained_weights):
        self.seqA_maxlen = 510
        self.maxlen = 512
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    def mask(self, seqA_tkn):
        masked_seqA_tkn = seqA_tkn.copy()
        tokenizer = self.tokenizer
        mask_token = tokenizer.mask_token
        seq_len = len(masked_seqA_tkn)
        n_repl = np.int(np.round(seq_len*0.15))  # number replaced
        ix_repl = np.random.choice(seq_len, n_repl)
        prob = np.random.rand(n_repl)
        for ix, p_ in zip(ix_repl, prob):
            if p_ <= 0.8:
                masked_seqA_tkn[ix] = mask_token
            elif 0.8 < p_ <= 0.9:
                masked_seqA_tkn[ix] = tokenizer.ids_to_tokens[np.random.randint(tokenizer.vocab_size)]
            elif p_ > 0.9:
                pass
        return masked_seqA_tkn

    def encode(self, seqA_list, init_ix):
        tokenizer = self.tokenizer
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        curr_ix = init_ix  # current index
        seqA_tkn = []
        while len(seqA_tkn) < self.seqA_maxlen and curr_ix < len(seqA_list):
            curr_tkn = tokenizer.tokenize(seqA_list[curr_ix])[:self.seqA_maxlen]
            if len(seqA_tkn) + len(curr_tkn) < self.seqA_maxlen:
                seqA_tkn += curr_tkn
                curr_ix += 1
            else:
                break
        masked_seqA_tkn = self.mask(seqA_tkn)
        seqA_tkn = [cls_token] + seqA_tkn + [sep_token]
        masked_seqA_tkn = [cls_token] + masked_seqA_tkn + [sep_token]
        tkn_list = seqA_tkn[:self.maxlen]
        masked_tkn_list = masked_seqA_tkn[:self.maxlen]
        attn_mask = [1] * len(tkn_list) + [0] * (self.maxlen - len(tkn_list))  # attn_mask for masked and non-masked tkn list should be equivalent
        tkn_list = tkn_list + [tokenizer.pad_token] * (self.maxlen - len(tkn_list))
        masked_tkn_list = masked_tkn_list + [tokenizer.pad_token] * (self.maxlen - len(masked_tkn_list))
        encoded = tokenizer.encode(tkn_list, add_special_tokens=False)
        masked_enc = tokenizer.encode(masked_tkn_list, add_special_tokens=False)
        return masked_enc, encoded, attn_mask

    def encode_batch(self, batch_seqA_list, batch_init_ix):
        batch_masked_enc = []
        batch_encoded = []
        batch_attn_mask = []
        for seqA, init_ix in zip(batch_seqA_list, batch_init_ix):
            masked_enc, encoded, attn_mask = self.encode(seqA, init_ix)
            batch_masked_enc.append(masked_enc)
            batch_encoded.append(encoded)
            batch_attn_mask.append(attn_mask)
        batch_masked_enc = torch.as_tensor(batch_masked_enc).long()
        batch_encoded = torch.as_tensor(batch_encoded).long()
        batch_attn_mask = torch.as_tensor(batch_attn_mask).long()
        return batch_masked_enc, batch_encoded, batch_attn_mask


# {"7.5M": {"steps_per_epoch": 50000, "batch_size": 10},
#  "30K": {"steps_per_epoch": 1000, "batch_size": 2}}
n_epochs = 15
steps_per_epoch = 50000  # 50000, 1000
batch_size_ = 10  # 10, 2
learn_rate_all = 1e-5
learn_rate_finetune = 2e-5
checkpoints_per_epoch = 5

model_params_dir = Path('K:/Work/ModelParams').expanduser()
model_trfmr_weights = str(model_params_dir / 'maskedlm_golbeck')

golbeck_dir = Path("K:/Work/Datasets-FakeNews/source-reliability/golbeck/FakeNewsData")
fakes_df_path = golbeck_dir / "fakes_df.tsv"
satires_df_path = golbeck_dir / "satires_df.tsv"
fakes_df = pd.read_csv(fakes_df_path, sep="\t").fillna("")
satires_df = pd.read_csv(satires_df_path, sep="\t").fillna("")

texts = np.hstack([[" ".join(row) for row in fakes_df[["title", "body"]].values],
                   [" ".join(row) for row in satires_df[["title", "body"]].values]])

tokenizer_class_ = DistilBertTokenizer
pretrained_weights_ = "distilbert-base-uncased"
mlm_tokenizer = MaskedLMTokenizer(tokenizer_class_, pretrained_weights_)
mlm = DistilBertForMaskedLM.from_pretrained(pretrained_weights_)
mlm.distilbert = mlm.distilbert.from_pretrained(pretrained_weights_)
mlm.to(device)

optimizer = AdamW([
    {"params": mlm.parameters(),
     "lr": learn_rate_finetune,
     "weight_decay": 1e-2}],
    lr=learn_rate_all,
    weight_decay=1e-3)
for i in range(n_epochs):
    train_epoch_loss = 0
    print("Epoch " + str(i+1))
    for step in np.arange(steps_per_epoch):
        mlm.train()
        optimizer.zero_grad()
        batch_doc_ix = np.random.randint(0, len(texts), batch_size_)
        batch_doc = [texts[ix] for ix in batch_doc_ix]
        batch_sent = [sent_tokenize(d) for d in batch_doc]
        batch_init_ix = [np.random.randint(0, l) for l in [len(s) for s in batch_sent]]
        batch_masked_enc_, batch_encoded_, batch_attn_mask_ = mlm_tokenizer.encode_batch(batch_sent, batch_init_ix)
        batch_masked_enc_ = torch.as_tensor(batch_masked_enc_).to(device)
        batch_encoded_ = torch.as_tensor(batch_encoded_).to(device)
        batch_attn_mask_ = torch.as_tensor(batch_attn_mask_).to(device)
        batch_loss, batch_pred = mlm(batch_masked_enc_, batch_attn_mask_, masked_lm_labels=batch_encoded_)[:2]
        batch_loss.backward()
        optimizer.step()
        train_epoch_loss += batch_loss.item()
        if (step + 1) % round(steps_per_epoch / checkpoints_per_epoch) == 0 or (step + 1) == steps_per_epoch:
            mlm.distilbert.save_pretrained(model_trfmr_weights)
            print("Epoch loss: " + str(round(train_epoch_loss / (step + 1), 4)))
