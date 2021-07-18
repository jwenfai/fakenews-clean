import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import dataset, RawField, Example, BucketIterator
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig, AdamW
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# import getpass
# user = getpass.getuser()
# if user == 'Low':
#     import os
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# else:
#     import os


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def similar_sents(claim, article, top_n=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    sents = sent_tokenize(article)
    tfidf_vecs = tfidf_vectorizer.fit_transform([claim] + sents)
    similarities = cosine_similarity(tfidf_vecs[0], tfidf_vecs[1:]).flatten()
    if not top_n:
        return [sents[ix] for ix in np.argsort(similarities)[::-1]]
    elif top_n:
        return [sents[ix] for ix in np.argsort(similarities)[::-1][:top_n]]


def prepare_text(text_title, text_body, output_type, max_body_sents):
    if output_type == "title-body":
        prepped_text = [[t] + sent_tokenize(b)[:max_body_sents] for t, b in zip(text_title, text_body)]
    elif output_type == "title-simsents":
        prepped_text = [[t] + similar_sents(t, b, max_body_sents) for t, b in zip(text_title, text_body)]
    return prepped_text


def get_weights(weight_name, weight_dir=None):
    weight_name_dict = dict(zip(
        ["simsents30k", "target30k", "external30k", "simsents7.5m", "target7.5m", "external7.5m"],
        ["maskedlm_golbeck_simsents_5k", "maskedlm_golbeck_5k", "maskedlm_nelagt_5k",
         "maskedlm_golbeck_simsents", "maskedlm_golbeck", "maskedlm_nelagt"]
    ))
    if weight_name == "distilbert-base-uncased":
        weight_path = weight_name
    else:
        assert weight_dir is not None
        weight_path = str(weight_dir / weight_name_dict[weight_name])
    return weight_path


class ModelTokenizer:
    def __init__(self, tokenizer_class, pretrained_weights):
        self.max_clm_len = 510
        self.max_sent_len = 62
        self.max_seq_len = 512
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    def encode(self, art_list):
        tokenizer = self.tokenizer
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        art_tkn_list = [[cls_token] + tokenizer.tokenize(art_)[:self.max_sent_len] + [sep_token] for art_ in art_list]
        art_tkn_list = [tkn for sublist in art_tkn_list for tkn in sublist][:self.max_seq_len][:-1] + [sep_token]
        attn_mask = [1] * len(art_tkn_list) + [0] * (self.max_seq_len - len(art_tkn_list))
        art_tkn_list = art_tkn_list + [tokenizer.pad_token] * (self.max_seq_len - len(art_tkn_list))
        encoded = tokenizer.encode(art_tkn_list, add_special_tokens=False)
        cls_loc = [ix for ix, tkn in enumerate(encoded) if tkn == tokenizer.cls_token_id]
        return encoded, attn_mask, cls_loc

    def encode_batch(self, batch_art_list):
        batch_encoded = []
        batch_attn_mask = []
        batch_cls_loc = []
        for art_list in batch_art_list:
            encoded, attn_mask, cls_loc = self.encode(art_list)
            batch_encoded.append(encoded)
            batch_attn_mask.append(attn_mask)
            batch_cls_loc.append(cls_loc)
        batch_encoded = torch.as_tensor(batch_encoded).long()
        batch_attn_mask = torch.as_tensor(batch_attn_mask).long()
        return batch_encoded, batch_attn_mask, batch_cls_loc


class SeqClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.3, rep_dim=768):
        super().__init__()
        self.rep_dim = rep_dim
        self.num_labels = num_labels
        self.seq_classif_dropout = dropout
        self.pre_classifier = nn.Linear(self.rep_dim, self.rep_dim)
        self.classifier = nn.Linear(self.rep_dim, self.num_labels)
        self.dropout = nn.Dropout(self.seq_classif_dropout)

    def forward(self, pooled_output):
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = F.log_softmax(logits, dim=1)
        return logits


class ClaimEvaluator(nn.Module):
    def __init__(self, num_labels, max_sents, pretrained_weights="distilbert-base-uncased"):
        super().__init__()
        self.concat_dim = 768 * (max_sents + 1)  # +1 for concatenated title sentence
        self.pretrained_weights = pretrained_weights
        self.num_labels = num_labels
        self.classifier = SeqClassifier(self.num_labels, rep_dim=self.concat_dim, dropout=0.1)
        self.bert_config = DistilBertConfig(dropout=0.1, attention_dropout=0.1)
        self.bert = DistilBertModel.from_pretrained(self.pretrained_weights, config=self.bert_config)

    def forward(self, art, art_attn, cls_tkn_list):
        pooled_output = self.bert(art, art_attn)[0]
        pooled_output = [output[tkn_loc].reshape(1, -1) for output, tkn_loc in zip(pooled_output, cls_tkn_list)]
        for ix, output in enumerate(pooled_output):
            output_len = len(output[0])
            if output_len < self.concat_dim:
                len_pad = self.concat_dim - output_len
                pooled_output[ix] = F.pad(output, (0, len_pad), "constant", 0)
        pooled_output = torch.cat(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


n_epochs = 10
n_classes = 2
batch_size_ = 8
learn_rate_all = 1e-5
learn_rate_finetune = 2e-5
checkpoints_per_epoch = 1
kfold_ = 10
max_sents_ = 16  # 16, False
text_type = "title-simsents"  # "title-body", "title-simsents"

pretrained_weights_name = "target7.5m"  # ["simsents", "target", "external"] * ["7.5m", "30k"], "distilbert-base-uncased"
model_params_dir = Path('K:/Work/ModelParams').expanduser()

golbeck_dir = Path("K:/Work/Datasets-FakeNews/source-reliability/golbeck/FakeNewsData")
fakes_df_path = golbeck_dir / "fakes_df.tsv"
satires_df_path = golbeck_dir / "satires_df.tsv"
fakes_df = pd.read_csv(fakes_df_path, sep="\t").fillna("")
satires_df = pd.read_csv(satires_df_path, sep="\t").fillna("")

fake_satire = (
    list(zip([1]*len(fakes_df), fakes_df["title"].values, fakes_df["body"].values)) +
    list(zip([0]*len(satires_df), satires_df["title"].values, satires_df["body"].values)))
random.shuffle(fake_satire)

fake_satire_X = [row[1] for row in fake_satire]
fake_satire_y = [row[0] for row in fake_satire]
skf = StratifiedKFold(n_splits=kfold_)
skf_splits = skf.split(fake_satire_X, fake_satire_y)

valid_metrics = []
for train_ix, valid_ix in skf_splits:
    print("")
    train_pairs = [fake_satire[ix] for ix in train_ix]
    valid_pairs = [fake_satire[ix] for ix in valid_ix]

    id_field = RawField()
    label_field = RawField()
    title_field = RawField()
    text_field = RawField()
    article_field = RawField()
    claim_fields = [('label', label_field), ('title', title_field), ('body', text_field)]
    train_examples = [Example.fromlist(row, claim_fields) for row in train_pairs]
    valid_examples = [Example.fromlist(row, claim_fields) for row in valid_pairs]
    train_dataset = dataset.Dataset(train_examples, claim_fields)
    valid_dataset = dataset.Dataset(valid_examples, claim_fields)

    model_trfmr_weights = get_weights(pretrained_weights_name, model_params_dir)
    tokenizer_class_ = DistilBertTokenizer
    tokenizer_weights_ = 'distilbert-base-uncased'
    model_tokenizer = ModelTokenizer(tokenizer_class_, tokenizer_weights_)
    claims_model = ClaimEvaluator(num_labels=n_classes, max_sents=max_sents_, pretrained_weights=model_trfmr_weights)
    claims_model.to(device)

    criterion = nn.NLLLoss().to(device)
    optimizer = AdamW([
        {"params": claims_model.classifier.parameters()},
        {"params": claims_model.bert.parameters(),
         "lr": learn_rate_finetune,
         "weight_decay": 1e-2}],
        lr=learn_rate_all,
        weight_decay=1e-3)
    valid_f1_hiscore = 0
    for i in range(n_epochs):
        train_pred_list = []
        train_tgt_list = []
        train_epoch_loss = 0
        print("Epoch " + str(i+1))
        train_iterator = BucketIterator(train_dataset, batch_size=batch_size_, shuffle=True)
        valid_iterator = BucketIterator(valid_dataset, batch_size=batch_size_, shuffle=False)
        for step, train_batch in enumerate(train_iterator):
            claims_model.train()
            optimizer.zero_grad()
            label_ = torch.as_tensor(train_batch.label).long().to(device)
            title_ = train_batch.title
            body_ = train_batch.body
            text_ = prepare_text(title_, body_, text_type, max_sents_)
            batch_encoded_, batch_attn_mask_, batch_cls_loc_, = model_tokenizer.encode_batch(text_)
            batch_encoded_ = torch.as_tensor(batch_encoded_).to(device)
            batch_attn_mask_ = torch.as_tensor(batch_attn_mask_).to(device)
            train_pred = claims_model(batch_encoded_, batch_attn_mask_, batch_cls_loc_)
            loss = criterion(train_pred, label_)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            train_pred_list.append(train_pred.detach().cpu().numpy())
            train_tgt_list.append(train_batch.label)
            if (step + 1) % round(len(train_iterator) / checkpoints_per_epoch) == 0 or (step + 1) == len(train_iterator):
                train_pred_scoring = np.argmax(np.vstack(train_pred_list), axis=1)
                train_tgt_scoring = np.hstack(train_tgt_list)
                train_f1 = f1_score(train_tgt_scoring, train_pred_scoring,
                                    labels=list(range(n_classes)), average='weighted')
                print("Training F1: " + str(train_f1.round(4)) + ", " +
                      "epoch loss: " + str(round(train_epoch_loss / (step+1), 4)))
                with torch.no_grad():
                    claims_model.eval()
                    valid_pred_list = []
                    valid_tgt_list = []
                    valid_epoch_loss = 0
                    for valid_batch in valid_iterator:
                        label_ = torch.as_tensor(valid_batch.label).long().to(device)
                        title_ = valid_batch.title
                        body_ = valid_batch.body
                        text_ = prepare_text(title_, body_, text_type, max_sents_)
                        batch_encoded_, batch_attn_mask_, batch_cls_loc_, = model_tokenizer.encode_batch(text_)
                        batch_encoded_ = torch.as_tensor(batch_encoded_).to(device)
                        batch_attn_mask_ = torch.as_tensor(batch_attn_mask_).to(device)
                        valid_pred = claims_model(batch_encoded_, batch_attn_mask_, batch_cls_loc_)
                        loss = criterion(valid_pred, label_)
                        valid_epoch_loss += loss.item()
                        valid_pred_list.append(valid_pred.detach().cpu().numpy())
                        valid_tgt_list.append(valid_batch.label)
                    valid_pred_flat = [pred for sublist in valid_pred_list for pred in sublist]
                    valid_pred_scoring = np.argmax(np.vstack(valid_pred_list), axis=1)
                    valid_tgt_scoring = np.hstack(valid_tgt_list)
                    valid_f1 = f1_score(valid_tgt_scoring, valid_pred_scoring,
                                        labels=list(range(n_classes)), average='weighted')
                    valid_acc = accuracy_score(valid_tgt_scoring, valid_pred_scoring,)
                    valid_loss = valid_epoch_loss / len(valid_iterator)
                    print("Validation F1: " + str(valid_f1.round(4)) + ", " +
                          "Acc: " + str(valid_acc.round(4)) + ", " +
                          "epoch loss: " + str(round(valid_loss, 4)))
                    if valid_f1 > valid_f1_hiscore:
                        valid_f1_hiscore = valid_f1
    valid_metrics.append([valid_f1, valid_acc, valid_loss])

print("\n".join(["\t".join([str(x) for x in row]) for row in valid_metrics]))
print("\t".join([str(x) for x in np.array(valid_metrics).mean(axis=0)]))
print("\t".join([str(x) for x in np.array(valid_metrics).std(axis=0)]))
