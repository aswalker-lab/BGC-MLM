# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 16:54:14 2025

@author: Allison Walker
"""
# adapted from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.compose import ColumnTransformer
import random
from pathlib import Path
import torch
import math
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import tqdm
from torch.utils.data import Dataset, DataLoader


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        if self.n_current_steps < self.n_warmup_steps:
            return np.power(self.n_warmup_steps, -1.7) * self.n_current_steps
        else:
            return np.min(
                [
                    np.power(self.n_current_steps, -0.45),
                    np.power(self.n_warmup_steps, -0.7),
                ]
            )
        # return np.min([
        #   np.power(self.n_current_steps, -0.2),
        #  np.power(self.n_warmup_steps, -1.2) * self.n_current_steps])

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        print("current steps: " + str(self.n_current_steps))
        print("lr: " + str(lr))
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class MLMTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_data,
        test_dataloader=None,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        log_freq=10,
        device="cpu",
    ):

        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bgc_mlm.d_model, n_warmup_steps=warmup_steps
        )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list = []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)

    def validate(self, validation_set, batch_size=2144):
        val_loader = DataLoader(validation_set, batch_size)
        print("validating...")
        data_iter = tqdm.tqdm(
            enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}"
        )
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # print(data)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                # print(data)
                mask_lm_output = self.model.forward(data["bert_input"])

                mask_loss = self.criterion(
                    mask_lm_output.transpose(1, 2), data["bert_label"]
                )

                avg_loss += mask_loss.item()

        return avg_loss / (i + 1)

    def predictSequenceMetrics(self):
        data_iter = tqdm.tqdm(
            enumerate(self.train_data),
            total=len(self.train_data),
            bar_format="{l_bar}{r_bar}",
            mininterval=1000,
        )
        # prediction_list = []
        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_5_accuracy = 0.0
        avg_10_accuracy = 0.0
        avg_max_val = 0.0
        self.model.eval()
        total_data_points = 0
        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # print(data)
                data = {key: value.to(self.device) for key, value in data.items()}

                predicted_index = torch.argmax(data["bert_label"])
                # skip if PAD or other special token
                if (
                    data["bert_label"][0][predicted_index] == 0
                    or data["bert_label"][0][predicted_index] == 1
                    or data["bert_label"][0][predicted_index] == 2
                ):
                    # print(data["bert_label"][0])
                    # print("HERE!!")
                    continue
                # 1. forward the next_sentence_prediction and masked_lm model
                # print(data)
                mask_lm_output = self.model.forward(data["bert_input"])
                # print(mask_lm_output)

                # 2-1. NLL(negative log likelihood) loss of is_next classification result
                # next_loss = self.criterion(next_sent_output, data["is_next"])

                # 2-2. NLLLoss of predicting masked token word
                # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
                # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
                mask_loss = self.criterion(
                    mask_lm_output.transpose(1, 2), data["bert_label"]
                )
                # print(mask_lm_output.transpose(1, 2))
                # print(data["bert_label"])
                # print("mask loss " + str(mask_loss))

                # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                # loss = next_loss + mask_loss
                loss = mask_loss
                avg_loss += mask_loss.item()
                # prediction_list.append(mask_lm_output.cpu().numpy())
                max_indices = torch.argmax(mask_lm_output, dim=2)
                max_vals, max_indices2 = torch.max(mask_lm_output, dim=2)
                max_val = max_vals[0][predicted_index].item()
                avg_max_val += max_val

                total_data_points += 1
                if (
                    max_indices[0][predicted_index]
                    == data["bert_label"][0][predicted_index]
                ):
                    avg_accuracy += 1
                # rank predictions at position for top 10 and top 50 metric
                top5_val, top5_ind = torch.topk(mask_lm_output, k=5, dim=2)
                top10_val, top10_ind = torch.topk(mask_lm_output, k=10, dim=2)

                if (
                    data["bert_label"][0][predicted_index]
                    in top5_ind[:, predicted_index, :]
                ):
                    avg_5_accuracy += 1
                if (
                    data["bert_label"][0][predicted_index]
                    in top10_ind[:, predicted_index, :]
                ):
                    avg_10_accuracy += 1

                # print("max indices")
                # print(max_indices)
                # torch.cuda.empty_cache()
        if total_data_points > 0:
            avg_loss = avg_loss / total_data_points
            avg_accuracy = avg_accuracy / total_data_points
            avg_5_accuracy = avg_5_accuracy / total_data_points
            avg_10_accuracy = avg_10_accuracy / total_data_points
            avg_max_val = avg_max_val / total_data_points
            print("average loss: " + str(avg_loss))
            print("avg accuracy: " + str(avg_accuracy))
            print("avg top 5: " + str(avg_5_accuracy))
            # return avg_loss, prediction_list, avg_accuracy, avg_5_accuracy, avg_10_accuracy, avg_max_val
            return avg_loss, avg_accuracy, avg_5_accuracy, avg_10_accuracy, avg_max_val
        else:
            # return None, prediction_list, None, None, None, None
            return None, None, None, None, None

    def encode(self, data_loader):
        # progress bar
        self.model.eval()
        encoded_list = []
        data_iter = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}"
        )
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # print(data)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            # print(data)
            mask_lm_output = self.model.forward(data["bert_input"])
            encoder_output1 = self.model.bert.forward(data["bert_input"])
            # print("HERE!")
            # print(encoder_output1.shape)
            # print(encoder_output1)
            np_output = encoder_output1.detach().cpu().numpy()
            np_output = np.copy(np_output)
            # print(np_output.shape)
            pos_average = np.average(np_output, axis=1)
            # print(pos_average.shape)
            encoded_list.append(pos_average)
            # np_output = np_output.flatten()
            # print("NP SIZE!")
            # print(np_output.size)
            # print(np_output[0].size)
            # print(np_output[0])

        return encoded_list

    def iteration(self, epoch, data_loader, val_data, train=True):

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            # print(data)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            # print(data)
            mask_lm_output = self.model.forward(data["bert_input"])
            # print(mask_lm_output)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            # next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion(
                mask_lm_output.transpose(1, 2), data["bert_label"]
            )

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            # loss = next_loss + mask_loss
            loss = mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)},\
                        val_loss={val_loss}"
            )
        else:
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)}"
            )

    def iterationPredict(self, epoch, data_loader, val_data, train=False):

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            # print(data)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            # print(data)
            mask_lm_output = self.model.forward(data["bert_input"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            # next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion(
                mask_lm_output.transpose(1, 2), data["bert_label"]
            )

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            # loss = next_loss + mask_loss
            loss = mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)},\
                        val_loss={val_loss}"
            )
        else:
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)}"
            )


class BGCMultiLabelTrainier:
    def __init__(
        self,
        model,
        train_dataloader,
        val_data,
        pos_weights,
        test_dataloader=None,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        log_freq=10,
        device="cpu",
    ):

        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data

        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.d_model, n_warmup_steps=warmup_steps
        )
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list = []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)

    def predict(self, data, batch_size=2144):
        data_loader = DataLoader(data, batch_size)
        data_iter = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}"
        )
        self.model.eval()
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                predictions = self.model.forward(data["bert_input"])
                activation = torch.nn.Sequential(torch.nn.Sigmoid())
                predictions = activation(predictions)
                if i == 0:
                    all_predictions = predictions
                else:
                    all_predictions = torch.cat((all_predictions, predictions))
        return all_predictions

    def validate(self, validation_set, batch_size=2144):
        val_loader = DataLoader(validation_set, batch_size)
        print("validating...")
        data_iter = tqdm.tqdm(
            enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}"
        )
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # print(data)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                # print(data)
                model_output = self.model.forward(data["bert_input"])

                loss = self.criterion(
                    model_output, data["classification_label"].float()
                )

                avg_loss += loss.item()

        return avg_loss / (i + 1)

    def iteration(self, epoch, data_loader, val_data, train=True):
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            model_output = self.model.forward(data["bert_input"])
            # if i == 0:
            #   print("model output ")
            #  print(model_output)
            # print("labels")
            # print(data["classification_label"])
            # 2-2. Calc loss
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            loss = self.criterion(model_output, data["classification_label"].float())

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)},\
                        val_loss={val_loss}"
            )
        else:
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)}"
            )


class BGCRegressionTrainier:
    def __init__(
        self,
        model,
        train_dataloader,
        val_data,
        test_dataloader=None,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        log_freq=10,
        device="cpu",
    ):

        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data

        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.d_model, n_warmup_steps=warmup_steps
        )
        self.criterion = torch.nn.MSELoss()
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list = []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)

    def predict(self, data, batch_size=2144):
        data_loader = DataLoader(data, batch_size)
        data_iter = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}"
        )
        self.model.eval()
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                predictions = self.model.forward(data["bert_input"])
                if i == 0:
                    all_predictions = predictions
                else:
                    all_predictions = torch.cat((all_predictions, predictions))
        return all_predictions

    def validate(self, validation_set, batch_size=2144):
        val_loader = DataLoader(validation_set, batch_size)
        print("validating...")
        data_iter = tqdm.tqdm(
            enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}"
        )
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # print(data)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                # print(data)
                model_output = self.model.forward(data["bert_input"])

                loss = self.criterion(model_output, data["regression_value"].float())

                avg_loss += loss.item()

        return avg_loss / (i + 1)

    def iteration(self, epoch, data_loader, val_data, train=True):
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            model_output = self.model.forward(data["bert_input"])
            # if i == 0:
            #   print("model output ")
            #  print(model_output)
            # print("labels")
            # print(data["classification_label"])
            # 2-2. Calc loss
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            loss = self.criterion(model_output, data["regression_value"].float())

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)},\
                        val_loss={val_loss}"
            )
        else:
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)}"
            )


class BGCMetricTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_data,
        test_dataloader=None,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        log_freq=10,
        device="cpu",
        loss_type="correlation",
    ):

        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data
        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.d_model, n_warmup_steps=warmup_steps
        )
        self.loss_type = loss_type
        if self.loss_type == "correlation":
            self.criterion = self.correlation_loss
        elif self.loss_type == "cross_entropy":
            self.criterion = self.cross_entropy_similarity_loss
        elif self.loss_type == "triplet":
            self.criterion = self.triplet_loss
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list = []

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)

    def predict(self, data, batch_size=2144):
        data_loader = DataLoader(data, batch_size)
        data_iter = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}"
        )
        self.model.eval()
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                predictions = self.model.forward(data["bert_input"])
                if i == 0:
                    all_predictions = predictions
                else:
                    all_predictions = torch.cat((all_predictions, predictions))
        return all_predictions

    def correlation_loss(self, d_embed, d_target, eps=1e-8, sum_diff_weight=0.1):
        # d_embed, d_target are shape (num_pairs,)
        d_embed_centered = d_embed - d_embed.mean()
        d_target_centered = d_target - d_target.mean()

        numerator = (d_embed_centered * d_target_centered).sum()
        denominator = torch.sqrt(
            (d_embed_centered**2).sum() * (d_target_centered**2).sum() + eps
        )

        corr = numerator / denominator
        # print("CORR")
        # print(corr)
        sum_difference = torch.sqrt(
            (
                d_embed.sum() / int(d_embed.size()[0])
                - d_target.sum() / int(d_target.size()[0])
            )
            ** 2
        )
        # print(d_embed.sum()/int(d_embed.size()[0]))
        # print(sum_difference)
        return 1 - corr  # + 0.1*sum_difference

    def cross_entropy_similarity_loss(self, z, target_sims, sigma=1.0, eps=1e-8):
        """
        z: (B, d)   - embedding vectors
        target_sims: (B, B) - target similarities (e.g., from cosine, kernels, etc.)
        Returns: scalar loss
        """
        device = z.device
        B = z.size(0)

        # --- pairwise squared distances in embedding space ---
        # diff: (B, B, d)
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        sqdist = (diff**2).sum(dim=-1)  # (B, B)

        # --- embedding similarities q_ij (Gaussian kernel) ---
        sim = torch.exp(-sqdist / (2.0 * sigma**2))  # (B, B)

        # make a mask for the diagonal (i == j)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)

        # zero out self-similarities WITHOUT in-place ops
        sim = sim.masked_fill(diag_mask, 0.0)

        # normalize rows to get q_ij
        row_sums_q = sim.sum(dim=1, keepdim=True) + eps
        q = sim / row_sums_q  # (B, B)

        # --- target probabilities p_ij from target_sims ---
        target_sims = target_sims.to(device).float()
        target_sims = target_sims.masked_fill(diag_mask, 0.0)

        row_sums_p = target_sims.sum(dim=1, keepdim=True) + eps
        p = target_sims / row_sums_p  # (B, B)

        # --- cross-entropy: sum_i sum_j p_ij * log(q_ij) ---
        # add eps inside log for numerical stability
        loss = -(p * (q + eps).log()).sum()

        return loss

    def triplet_loss(self, anchor, pos, neg, margin=0.1):
        d_pos = ((anchor - pos) ** 2).sum(dim=-1).sqrt()
        d_neg = ((anchor - neg) ** 2).sum(dim=-1).sqrt()

        return torch.relu(d_pos - d_neg + margin).mean()

    # Given target distance matrix D (N, N)
    def sample_triplets_indices(
        self, D, num_triplets, tau_pos=0.2, tau_neg=0.2, eps=1e-12
    ):
        """
        Sample (anchor, positive, negative) index triplets using a target distance matrix.

        D: (N, N) target distance matrix (smaller = more similar)
        num_triplets: number of triplets to sample
        tau_pos: temperature for positive sampling (smaller -> more strongly favors nearest)
        tau_neg: temperature for negative sampling (smaller -> more strongly favors farthest)
        Returns:
            a, p, n: Long tensors of shape (num_triplets,)
        """
        device = D.device
        N = D.shape[0]

        # sample anchors uniformly
        a = torch.randint(0, N, (num_triplets,), device=device)

        # gather anchor rows
        rows = D[a]  # (T, N)

        # ---- POSITIVES: prefer small distances ----
        tau_pos = max(float(tau_pos), 1e-6)
        pos_logits = -rows / tau_pos

        # mask self (i == j) in logits
        pos_logits[torch.arange(num_triplets, device=device), a] = -float("inf")

        pos_probs = torch.softmax(pos_logits, dim=1)
        p = torch.multinomial(pos_probs, 1).squeeze(1)

        # ---- NEGATIVES: prefer large distances ----
        tau_neg = max(float(tau_neg), 1e-6)
        neg_logits = rows / tau_neg

        # mask self again
        neg_logits[torch.arange(num_triplets, device=device), a] = -float("inf")

        neg_probs = torch.softmax(neg_logits, dim=1)
        n = torch.multinomial(neg_probs, 1).squeeze(1)

        return a.long(), p.long(), n.long()

    def validate(self, validation_set, batch_size=2144):
        val_loader = DataLoader(validation_set, batch_size)
        print("validating...")
        data_iter = tqdm.tqdm(
            enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}"
        )
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # print(data)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                # print(data)
                model_output = self.model.forward(data["bert_input"])
                if self.loss_type == "correlation":
                    d_embed = pairwise_cosine_vector(model_output)
                    d_target = pairwise_cosine_vector(data["target_embedding"].float())
                    loss = self.criterion(d_embed, d_target)
                elif self.loss_type == "cross_entropy":
                    sim = cosine_similarity_matrix(data["target_embedding"].float())
                    loss = self.criterion(model_output, sim)
                elif self.loss_type == "triplet":
                    d_target = cosine_distance_matrix(data["target_embedding"].float())
                    a, p, n = self.sample_triplets_indices(
                        d_target, 2048, tau_pos=0.1, tau_neg=0.1
                    )

                    loss = self.criterion(
                        model_output[a], model_output[p], model_output[n], margin=0.1
                    )

                avg_loss += loss.item()

        return avg_loss / (i + 1)

    def iteration(self, epoch, data_loader, val_data, train=True):
        avg_loss = 0.0
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            model_output = self.model.forward(data["bert_input"])
            if self.loss_type == "correlation":
                # calculate distances
                d_embed = pairwise_cosine_vector(model_output)
                d_target = pairwise_cosine_vector(data["target_embedding"].float())
                # print()
                # print(d_embed)
                # print(d_target)
                # print()
                # 2-2. Calc loss
                # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
                # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
                loss = self.criterion(d_embed, d_target)
            elif self.loss_type == "cross_entropy":
                sim = cosine_similarity_matrix(data["target_embedding"].float())
                loss = self.criterion(model_output, sim)
            elif self.loss_type == "triplet":
                d_target = cosine_distance_matrix(data["target_embedding"].float())
                a, p, n = self.sample_triplets_indices(
                    d_target, 2048, tau_pos=0.1, tau_neg=0.1
                )

                loss = self.criterion(
                    model_output[a], model_output[p], model_output[n], margin=0.1
                )

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)},\
                        val_loss={val_loss}"
            )
        else:
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)}"
            )
