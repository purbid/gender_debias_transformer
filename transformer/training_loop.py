import sys
import json
import spacy
import torch
import math
import torch.nn as nn
import torch.optim as optim
from random import randrange
from nltk.corpus import stopwords
from transformer_utils import pad
from transformer_class import Transformer
from torch .utils.tensorboard import SummaryWriter
stop_words = stopwords.words('english')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True
save_model = True


class Trainer():

    def __init__(self,
                 learning_rate,
                 dict_file,
                 SAVE_MODEL_PATH,
                 FEATURE_SEP_TRAIN_DATA,
                 ):

        self.model = Transformer(dict_file)
        self.data = FEATURE_SEP_TRAIN_DATA
        self.save_to = SAVE_MODEL_PATH
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=10, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.src_pad_idx)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.src_pad_idx)
        self.start_id = self.model.dict.tok2ind[self.model.dict.start_token]
        self.null_id = self.model.dict.tok2ind[self.model.dict.null_token]

        self.eos_id = self.model.dict.tok2ind[self.model.dict.end_token]
        self.unk_id = self.model.dict.tok2ind[self.model.dict.unk_token]
        self.vocab_size = len(self.model.dict)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduce=True, reduction='mean')
        self.padding_weights = torch.ones(self.vocab_size).to(device)
        self.padding_weights[self.null_id] = 0
        self.WeightedCrossEntropyLoss = nn.CrossEntropyLoss(weight=self.padding_weights, reduce=True, reduction='mean')

        #############MAKE SURE TO ADD GPU LINE AS WELL.
        if load_model:
            self.load_checkpoint(torch.load(SAVE_MODEL_PATH, map_location=torch.device('cpu')), self.model, self.optimizer)

    def save_checkpoint(self, state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint, optimizer):
        print("=> Loading checkpoint")
        self.model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    def accuracy(self, predictions, truths):
        print("Here to calculate the accuracy of the predictions")

        # for i,j in zip(predictions, truths):
        #     print("prediction is: "+str([model.dict.ind2tok.get(index) for index in i.tolist()]))
        #     print("truth is: "+str([model.dict.ind2tok.get(index) for index in j.tolist()]))
        #     print("\n")
        #
        # exit()
        ###### only where
        pads = torch.Tensor.double(truths != 0)
        corrects = torch.Tensor.double(predictions == truths)
        valid_corrects = corrects * pads

        return valid_corrects.sum() / pads.sum()

    def eval_loss(self, dists, text, pad_token):
        print("here to calculate eval loss, different than training loss")

        loss = 0
        num_tokens = 0

        for dist, y in zip(dists, text):
            y_len = sum([1 if y_i != pad_token else 0 for y_i in y])
            for i in range(y_len):
                loss -= torch.log(dist[i][y[i]])
                num_tokens += 1

        return loss, num_tokens

    def prepare_training_datasets(self, batch_size, n=10):
        with open(self.data, 'r') as f:
            data_list = json.load(f)
            train_data_list = data_list[:-4500]
            valid_data_list = data_list[-4500:]
        train_data = []
        ######range(5000//321):
        ###first batch is 0:320
        for i in range(len(train_data_list) // (n * batch_size) + 1):
            nbatch = train_data_list[i * (n * batch_size): (i + 1) * (n * batch_size)]
            nbatch_list = [([self.model.dict.tok2ind.get(word, self.unk_id) for word in
                             [self.model.dict.start_token, *(ins[0].split()), self.model.dict.end_token]],
                            ins[1]) for ins in nbatch]
            descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)
            j = 0

            #####descend_nbatch_list is the wordtoid descending order in terms of num of words
            #### making mini batches of size 32
            while len(descend_nbatch_list[j * batch_size: (j + 1) * batch_size]) > 0:
                batch_list = descend_nbatch_list[j * batch_size: (j + 1) * batch_size]
                # text: (batch_size x seq_len)
                ########## this pads the extra spacew ith zeroes

                text = pad([x[0] for x in batch_list], padding=self.null_id)
                labels = torch.tensor([x[1] for x in batch_list], dtype=torch.long)

                train_data.append((text, labels))
                j += 1

        """
        example of train data tensor([   8,  175,  111,   11,  236,   18,  626,   22, 2000,    3,    9,   31,
                  28, 3094,   82, 5763,  115,   17, 1263,    4]) tensor(0)
        """

        valid_data = []
        for i in range(len(valid_data_list) // (n * batch_size) + 1):
            nbatch = valid_data_list[i * (n * batch_size): (i + 1) * (n * batch_size)]

            nbatch_list = [([self.model.dict.tok2ind.get(word, self.unk_id) for word in
                             [self.model.dict.start_token, *(ins[0].split()), self.model.dict.end_token]],
                            ins[1]) for ins in nbatch]
            descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)

            j = 0
            while len(descend_nbatch_list[j * batch_size: (j + 1) * batch_size]) > 0:
                batch_list = descend_nbatch_list[j * batch_size: (j + 1) * batch_size]

                # text: (batch_size x seq_len)
                text = pad([x[0] for x in batch_list], padding=self.null_id)

                labels = torch.tensor([x[1] for x in batch_list], dtype=torch.long)

                valid_data.append((text, labels))
                j += 1

        return train_data, valid_data



    def train(self, n_epoch):

        step = 0
        losses = []

        self.train_data, self.valid_data = self.prepare_training_datasets(self.model, 32)

        for epoch in range(n_epoch):
            for i_batch, batch in enumerate(self.train_data):
                # text: (batch_size x seq_len)
                # labels: (batch_size)

                text, target = batch

                text = text.to(device)
                target = target.to(device)

                self.model.train()
                output = self.model(text, target[:-1, :])
                output = output.reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)
                self.optimizer.zero_grad()
                reconstruction_loss = self.WeightedCrossEntropyLoss(output, target)
                losses.append(reconstruction_loss.item())
                reconstruction_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()


                if torch.isnan(loss):
                    print("loss NAN.")
                    print(reconstruction_loss)
                    continue




                text = text.permute(1, 0)
                text = text[1:]
                re_acc = self.accuracy(preds[1:], text)


                if i_batch % 100 == 0:
                    print(
                        "train autoencoder epoch: {}  batch: {} / {}  batch_size: {}  loss: {:.6f}  re_acc: {:.4f} ".format(
                            epoch, i_batch, len(self.train_data), len(text), reconstruction_loss.item(), re_acc))
                if i_batch % 1000 == 0:
                    print("-----------------------------Validation-----------------------------")
                    self.model.eval()
                    total_loss = 0
                    total_num_tokens = 0

                    #
                    r = 0

                    for i_batch, batch in enumerate(self.valid_data):
                        # text: (batch_size x seq_len)
                        # labels: (batch_size)
                        val_text, val_labels = batch
                        val_text = val_text.to(device)
                        val_labels = val_labels.to(device)

                        # preds: (batch_size x seq_len)
                        # scores: (batch_size x seq_len x vocab_size)

                        val_text = val_text.permute(1, 0)
                        target = val_text.detach().clone().to(device)
                        output = self.model(val_text,target, val=True)


                        loss, num_tokens = self.eval_loss(output[1:], target[1:], pad_token=self.null_id)
                        total_loss += loss.item()
                        total_num_tokens += num_tokens

                        re_acc = self.accuracy(preds[1:], val_text[1:])

                    if i_batch == r:
                        val_text = val_text.permute(1, 0)
                        preds = preds.permute(1, 0)
                        for truth, pred in zip(val_text, preds):
                            truth_null_id_list = [i for i in range(len(truth)) if truth[i] == self.null_id]
                            pred_eos_id_list = [i for i in range(len(pred)) if pred[i] == self.eos_id]

                            truth_null_id = len(truth) if len(truth_null_id_list) == 0 else truth_null_id_list[0]
                            # print([model.dict.ind2tok.get(x) for x in truth.tolist()])
                            # print([model.dict.ind2tok.get(x) for x in pred.tolist()])

                            pred_eos_id = len(pred) if len(pred_eos_id_list) == 0 else pred_eos_id_list[0]
                            print('truth: {}'.format(
                                ' '.join([self.model.dict.ind2tok.get(idx, '__unk__') for idx in
                                          truth[1:truth_null_id].tolist()])))
                            print('pred: {}'.format(
                                ' '.join([self.model.dict.ind2tok.get(idx, '__unk__') for idx in
                                          pred[1:pred_eos_id].tolist()])))
                            print('------------------------------------------------------------')
                ave_loss = total_loss / total_num_tokens
                ppl = math.exp(ave_loss)

                torch.save(self.model.state_dict(),self.save_to)
                print("Model saved to save_model/disen_model.pt")
                print("Model performance: ppl_{:.4f} ".format(ppl))

#
# src = torch.rand(64, 16, 512)
# tgt = torch.rand(64, 16, 512)
# model = Transformer()
# out = model(src, tgt)


if __name__ == "__main__":
    print("yes")