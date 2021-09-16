import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import json
import sys
from random import randrange
from transformer.transformer_utils import pad
from torch .utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator, TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from disentangle_model.dict_v0 import DictionaryAgent

################ needed only once #######################
with open('/Users/purbid/PycharmProjects/debias/Debiased-Chat/debias/data/adv_train_data.json', 'r') as f:
    train_gender_data_list, train_neutral_data_list = json.load(f)
total_data=[]
total_data.extend(train_gender_data_list)
total_data.extend(train_neutral_data_list)

l=int(len(total_data)*0.7)
total_train_data=total_data[:l]
total_val_data=total_data[l:]

print("starting sort")
##### sort by size, so similar length examples are together #####
import time
tt=time.time()
total_train_data = sorted(total_train_data, key=lambda x: len(x[0]))
total_val_data = sorted(total_val_data, key=lambda x: len(x[0]))

print(time.time()-tt)
print("sorted by dialogue length")







# raw_data={'dialogue':[x[0] for x in total_data], "response":[x[1] for x in total_data]}
# df = pd.DataFrame(raw_data, columns=["dialogue","response"])
# train, val = train_test_split(df, test_size=0.1)
# train.to_csv("train_dialogue.csv", index=False)
# val.to_csv("val_dialogue.csv", index=False)
################ needed only once #######################

# twitter_text = Field(init_token = "<sos>", eos_token = "<eos>")
# data_fields = [('dialogue', twitter_text), ('response', twitter_text)]
# train,val = TabularDataset.splits(path='./', train='train_dialogue.csv', validation='val_dialogue.csv', format='csv', fields=data_fields)
# print("building_vocab ...")
# twitter_text.build_vocab(train, max_size=30000, min_freq=2)
# print("done building vocab ...")

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        dict_file,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):

        super(Transformer, self).__init__()
        dict_opt = {'dict_file': dict_file}
        self.dict = DictionaryAgent(dict_opt)
        self.src_word_embedding = nn.Embedding(len(self.dict), embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(len(self.dict), embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.NULL_IDX = self.dict.tok2ind[self.dict.null_token]
        self.START_IDX = self.dict.tok2ind[self.dict.start_token]
        self.END_IDX = self.dict.tok2ind[self.dict.end_token]
        self.src_pad_idx = self.dict.tok2ind[self.dict.null_token]
        self.longest_label = 30

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, len(self.dict))
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(device)

    def forward(self, src, trg):

        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
                .unsqueeze(1)
                .expand(src_seq_length, N)
                .to(device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_length)
                .unsqueeze(1)
                .expand(trg_seq_length, N)
                .to(device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = True
save_model = True
SAVE_MODEL_PATH = ""

# Training hyperparameters
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters

embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
dict_file="/Users/purbid/PycharmProjects/debias/Debiased-Chat/debias/data/twitter_seq2seq_model.dict"
# src_pad_idx = twitter_text.vocab.stoi["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

model = Transformer(
    embedding_size,
    dict_file,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)


n=10
batch_size = 32


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"] )
    optimizer.load_state_dict(checkpoint["optimizer"])





null_id=model.dict.tok2ind[model.dict.null_token]
final_train_data=[]

for i in range(len(total_train_data) // (n * batch_size) + 1):
    nbatch = total_train_data[i * (n * batch_size) : (i+1) * (n * batch_size)]
    nbatch_list = [([model.dict.tok2ind.get(word, model.dict.tok2ind.get("__unk__")) for word in
                     [model.dict.start_token, *(ins[0].split()), model.dict.end_token]],
                    [model.dict.tok2ind.get(word, model.dict.tok2ind.get("__unk__")) for word in
                     [model.dict.start_token, *(ins[1].split()), model.dict.end_token]]
                    , ins[2]) for ins in nbatch]

    descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)
    j = 0

    #####descend_nbatch_list is the word to id descending order in terms of num of words
    #### making mini batches of size 32
    while len(descend_nbatch_list[j * batch_size : (j+1) * batch_size]) > 0:
        batch_list = descend_nbatch_list[j * batch_size : (j+1) * batch_size]
        # text: (batch_size x seq_len)
        ########## this pads the extra spacew ith zeroes


        padded_dialogue = pad([x[0] for x in batch_list], padding=null_id)
        padded_response = pad([x[1] for x in batch_list], padding=null_id)
        labels = torch.tensor([x[2] for x in batch_list], dtype=torch.long)


        # for i, j in zip(padded_dialogue, padded_response):
        #     print("dialogue")
        #     print([model.dict.ind2tok.get(index_to_token) for index_to_token in i.tolist() ])
        #     print("response")
        #     print([model.dict.ind2tok.get(index_to_token) for index_to_token in j.tolist() ])
        final_train_data.append((padded_dialogue, padded_response, labels))
        j += 1




optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)
criterion = nn.CrossEntropyLoss(ignore_index=model.src_pad_idx)
if load_model:
    load_checkpoint(torch.load(SAVE_MODEL_PATH, map_location=torch.device('cpu')), model, optimizer)



for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    for _ in range(1):
        random_val_eg=randrange(len(total_val_data))
        sentence = total_val_data[random_val_eg][0]
        translated_sentence = translate_sentence(
            model, "do you think a female president will be good for the country", device, max_length=50
        )
        print(f"Response predicted: \n {translated_sentence}")
        print("response should be(gorund truth): "+str(total_val_data[random_val_eg][1]))
        print("\n")
        print("----------------")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(final_train_data[:5]):

        (inp_data, target, gender_label) = batch
        inp_data = inp_data.permute(1, 0).to(device)
        target = target.permute(1, 0).to(device)


        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        print("loss is "+str(loss.item()))
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)



