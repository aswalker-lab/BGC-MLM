import torch
import numpy as np
from torch.optim import AdamW
import tqdm
from torch.utils.data import DataLoader
from src.models.factory import pairwise_cosine_vector, cosine_similarity_matrix, cosine_distance_matrix

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

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
            return np.min([np.power(self.n_current_steps, -0.45), np.power(self.n_warmup_steps,-0.7)])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        print("current steps: " + str(self.n_current_steps))
        print("lr: " + str(lr))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class MLMTrainer:
    def __init__(self, model, train_dataloader, val_data, test_dataloader=None, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), warmup_steps=1000, log_freq=10, device='cpu'):
        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data
        
        # Determine d_model (wrapper vs raw)
        d_model = self.model.bgc_mlm.d_model if hasattr(self.model, "bgc_mlm") else self.model.d_model
        
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, d_model, n_warmup_steps=warmup_steps)
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list= []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)
    
    def validate(self, validation_set, batch_size=2144):
        val_loader = DataLoader(validation_set, batch_size)
        print('validating...')
        data_iter = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                mask_lm_output = self.model.forward(data["bert_input"])
                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                avg_loss += mask_loss.item()
        return avg_loss / (i + 1)
    
    def predictSequenceMetrics(self):
        data_iter = tqdm.tqdm(enumerate(self.train_data), total=len(self.train_data), bar_format="{l_bar}{r_bar}", mininterval=1000)
        avg_loss, avg_accuracy, avg_5_accuracy, avg_10_accuracy, avg_max_val = 0.0, 0.0, 0.0, 0.0, 0.0
        self.model.eval()
        total_data_points = 0
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                predicted_index = torch.argmax(data["bert_label"])
                if data["bert_label"][0][predicted_index] in [0, 1, 2]:
                    continue
                mask_lm_output = self.model.forward(data["bert_input"])
                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                avg_loss += mask_loss.item()
                max_indices = torch.argmax(mask_lm_output,dim=2)
                max_vals, max_indices2 = torch.max(mask_lm_output,dim=2)
                max_val = max_vals[0][predicted_index].item()
                avg_max_val += max_val
                total_data_points += 1
                if max_indices[0][predicted_index] == data["bert_label"][0][predicted_index]:
                    avg_accuracy += 1
                top5_val, top5_ind = torch.topk(mask_lm_output, k=5, dim=2)
                top10_val, top10_ind = torch.topk(mask_lm_output, k=10, dim=2)
                if data["bert_label"][0][predicted_index] in top5_ind[:,predicted_index,:]:
                    avg_5_accuracy += 1
                if data["bert_label"][0][predicted_index] in top10_ind[:,predicted_index,:]:
                    avg_10_accuracy += 1
        if total_data_points > 0:
            avg_loss /= total_data_points
            avg_accuracy /= total_data_points
            avg_5_accuracy /= total_data_points
            avg_10_accuracy /= total_data_points
            avg_max_val /= total_data_points
            print("average loss: " + str(avg_loss))
            print("avg accuracy: " + str(avg_accuracy))
            print("avg top 5: " + str(avg_5_accuracy))
            return avg_loss, avg_accuracy, avg_5_accuracy, avg_10_accuracy, avg_max_val
        return None,  None, None, None, None
    
    def encode(self, data_loader):
        self.model.eval()
        encoded_list = []
        data_iter = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            # handle both bare models and MultiLabel versions
            if hasattr(self.model, "bert"):
                encoder_output1 = self.model.bert.forward(data["bert_input"])
            elif hasattr(self.model, "bgc_mlm"):
                encoder_output1 = self.model.bgc_mlm(data["bert_input"])
            else:
                 encoder_output1 = self.model(data["bert_input"])
            np_output = encoder_output1.detach().cpu().numpy()
            np_output = np.copy(np_output)
            pos_average = np.average(np_output,axis=1)
            encoded_list.append(pos_average)
        return encoded_list
        
    def iteration(self, epoch, data_loader, val_data, train=True):
        avg_loss = 0.0
        mode = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (mode, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            mask_lm_output = self.model.forward(data["bert_input"])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            loss = mask_loss
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            avg_loss += loss.item()
            post_fix = {"epoch": epoch, "iter": i, "avg_loss": avg_loss / (i + 1), "loss": loss.item()}
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}, val_loss={val_loss}")    
        else:
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}")    
    
    def iterationPredict(self, epoch, data_loader, val_data, train=False):
        self.iteration(epoch, data_loader, val_data, train)


class BGCMultiLabelTrainier():
    def __init__(self, model, train_dataloader, val_data, pos_weights, test_dataloader=None, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), warmup_steps=1000, log_freq=10, device='cpu'):
        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data
        
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.d_model, n_warmup_steps=warmup_steps)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list= []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)
    
    def predict(self,data, batch_size=2144):
        data_loader = DataLoader(data, batch_size)
        data_iter = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}")
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
        print('validating...')
        data_iter = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                model_output = self.model.forward(data["bert_input"])
                loss = self.criterion(model_output, data["classification_label"].float())
                avg_loss += loss.item()
        return avg_loss / (i + 1)

    def iteration(self, epoch, data_loader, val_data, train=True):
        avg_loss = 0.0
        mode = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (mode, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            model_output = self.model.forward(data["bert_input"])
            loss = self.criterion(model_output, data["classification_label"].float())

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()
            post_fix = {"epoch": epoch, "iter": i, "avg_loss": avg_loss / (i + 1), "loss": loss.item()}
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}, val_loss={val_loss}")    
        else:
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}")    


class BGCRegressionTrainier():
    def __init__(self, model, train_dataloader, val_data, test_dataloader=None, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), warmup_steps=1000, log_freq=10, device='cpu'):
        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data
        
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.d_model, n_warmup_steps=warmup_steps)
        self.criterion = torch.nn.MSELoss()
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list= []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)
    
    def predict(self,data, batch_size=2144):
        data_loader = DataLoader(data, batch_size)
        data_iter = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}")
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
        print('validating...')
        data_iter = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                model_output = self.model.forward(data["bert_input"])
                loss = self.criterion(model_output, data["regression_value"].float())
                avg_loss += loss.item()
        return avg_loss / (i + 1)

    def iteration(self, epoch, data_loader, val_data, train=True):
        avg_loss = 0.0
        mode = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (mode, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            model_output = self.model.forward(data["bert_input"])
            loss = self.criterion(model_output, data["regression_value"].float())

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            avg_loss += loss.item()
            post_fix = {"epoch": epoch, "iter": i, "avg_loss": avg_loss / (i + 1), "loss": loss.item()}
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}, val_loss={val_loss}")    
        else:
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}") 

class BGCMetricTrainer():
    def __init__(self, model, train_dataloader, val_data, test_dataloader=None, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), warmup_steps=1000, log_freq=10, device='cpu', loss_type='correlation'):
        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.d_model, n_warmup_steps=warmup_steps)
        self.loss_type = loss_type
        if self.loss_type == "correlation":
            self.criterion = self.correlation_loss
        elif self.loss_type == "cross_entropy":
            self.criterion = self.cross_entropy_similarity_loss
        elif self.loss_type == "triplet":
            self.criterion = self.triplet_loss
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list= []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)
        
    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)
        
    def predict(self,data, batch_size=2144):
        data_loader = DataLoader(data, batch_size)
        data_iter = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), bar_format="{l_bar}{r_bar}")
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
    
    def correlation_loss(self, d_embed, d_target,eps=1e-8,sum_diff_weight=0.1):
        d_embed_centered = d_embed - d_embed.mean()
        d_target_centered = d_target - d_target.mean()
        numerator = (d_embed_centered * d_target_centered).sum()
        denominator = torch.sqrt((d_embed_centered**2).sum() * (d_target_centered**2).sum() + eps)
        corr = numerator / denominator
        sum_difference = torch.sqrt((d_embed.sum()/int(d_embed.size()[0])-d_target.sum()/int(d_target.size()[0]))**2)
        return (1 - corr)
    
    def cross_entropy_similarity_loss(self, z, target_sims, sigma=1.0, eps=1e-8):
        device = z.device
        B = z.size(0)
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        sqdist = (diff ** 2).sum(dim=-1) 
        sim = torch.exp(-sqdist / (2.0 * sigma ** 2))
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(diag_mask, 0.0)
        row_sums_q = sim.sum(dim=1, keepdim=True) + eps
        q = sim / row_sums_q
        target_sims = target_sims.to(device).float()
        target_sims = target_sims.masked_fill(diag_mask, 0.0)
        row_sums_p = target_sims.sum(dim=1, keepdim=True) + eps
        p = target_sims / row_sums_p
        loss = -(p * (q + eps).log()).sum()
        return loss
    
    def triplet_loss(self, anchor, pos, neg, margin=0.1):
        d_pos = ((anchor - pos)**2).sum(dim=-1).sqrt()
        d_neg = ((anchor - neg)**2).sum(dim=-1).sqrt()
        return torch.relu(d_pos - d_neg + margin).mean()
    
    def sample_triplets_indices(self, D, num_triplets, tau_pos=0.2, tau_neg=0.2, eps=1e-12):
        device = D.device
        N = D.shape[0]
        a = torch.randint(0, N, (num_triplets,), device=device)
        rows = D[a]
        tau_pos = max(float(tau_pos), 1e-6)
        pos_logits = -rows / tau_pos
        pos_logits[torch.arange(num_triplets, device=device), a] = -float("inf")
        pos_probs = torch.softmax(pos_logits, dim=1)
        p = torch.multinomial(pos_probs, 1).squeeze(1)
        tau_neg = max(float(tau_neg), 1e-6)
        neg_logits = rows / tau_neg
        neg_logits[torch.arange(num_triplets, device=device), a] = -float("inf")
        neg_probs = torch.softmax(neg_logits, dim=1)
        n = torch.multinomial(neg_probs, 1).squeeze(1)
        return a.long(), p.long(), n.long()
    
    def validate(self, validation_set, batch_size=2144):
        val_loader = DataLoader(validation_set, batch_size)
        print('validating...')
        data_iter = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                model_output = self.model.forward(data["bert_input"])
                if self.loss_type == "correlation":
                    d_embed = pairwise_cosine_vector(model_output)
                    d_target = pairwise_cosine_vector(data["target_embedding"].float())
                    loss = self.criterion(d_embed, d_target)
                elif self.loss_type == "cross_entropy":
                    sim = cosine_similarity_matrix(data["target_embedding"].float())
                    loss = self.criterion(model_output,sim)
                elif self.loss_type == "triplet":
                    d_target = cosine_distance_matrix(data["target_embedding"].float())
                    a, p, n = self.sample_triplets_indices(d_target, 2048, tau_pos=0.1, tau_neg=0.1)
                    loss = self.criterion(model_output[a], model_output[p], model_output[n], margin=0.1)
            
                avg_loss += loss.item()
        return avg_loss / (i + 1)
    
    def iteration(self, epoch, data_loader, val_data, train=True):
        avg_loss = 0.0
        mode = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (mode, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            model_output = self.model.forward(data["bert_input"])
            if self.loss_type == "correlation":
                d_embed = pairwise_cosine_vector(model_output)
                d_target = pairwise_cosine_vector(data["target_embedding"].float())
                loss = self.criterion(d_embed, d_target)
            elif self.loss_type == "cross_entropy":
                sim = cosine_similarity_matrix(data["target_embedding"].float())
                loss = self.criterion(model_output,sim)
            elif self.loss_type == "triplet":
                d_target = cosine_distance_matrix(data["target_embedding"].float())
                a, p, n = self.sample_triplets_indices(d_target, 2048, tau_pos=0.1, tau_neg=0.1)
                loss = self.criterion(model_output[a], model_output[p], model_output[n], margin=0.1)
            
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()
            post_fix = {"epoch": epoch, "iter": i, "avg_loss": avg_loss / (i + 1), "loss": loss.item()}
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}, val_loss={val_loss}")    
        else:
            print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}") 
