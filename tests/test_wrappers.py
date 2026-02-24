import unittest
import torch
import sys
import os
import random

# Ensure the BGC_MLM_tools and src are correctly importable for comparisons
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BGC-MLM-old'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import BGC_MLM_tools
import src.models.wrappers as model_wrappers
import src.data.wrappers as data_wrappers

class TestWrappers(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.seq_len = 50
        self.d_model = 64
        self.n_layers = 2
        self.heads = 4
        self.dropout = 0.1
        self.token_list = ["PAD", "UNK", "CLS", "SEP", "MASK"] + [f"tok{i}" for i in range(5, 100)]
        
        # Pad dummy sequences to seq_len
        dummy_seq = ["CLS", "tok5", "tok6", "tok7", "SEP"]
        dummy_seq += ["PAD"] * (self.seq_len - len(dummy_seq))
        self.bgc_tokens = [dummy_seq]
        
        self.y_vals = [[0, 1, 0]]
        self.mask_inds = [1]
        
    def test_mlm_dataset_parity(self):
        random.seed(42)
        old_ds = BGC_MLM_tools.MLMDataset(self.bgc_tokens, self.token_list, self.seq_len)
        old_item = old_ds[0]
        
        random.seed(42)
        new_ds = data_wrappers.MLMDataset(self.bgc_tokens, self.token_list, self.seq_len)
        new_item = new_ds[0]
        
        torch.testing.assert_close(old_item["bert_input"], new_item["bert_input"])
        torch.testing.assert_close(old_item["bert_label"], new_item["bert_label"])

    def test_model_parity(self):
        torch.manual_seed(42)
        old_bgc_mlm = BGC_MLM_tools.BGC_MLM(
            vocab_size=self.vocab_size, seq_len=self.seq_len, d_model=self.d_model, 
            n_layers=self.n_layers, heads=self.heads, dropout=self.dropout
        )
        old_mlm = BGC_MLM_tools.MLM(old_bgc_mlm, self.vocab_size)
        
        torch.manual_seed(42)
        new_mlm = model_wrappers.create_mlm_model(
            vocab_size=self.vocab_size, seq_len=self.seq_len, d_model=self.d_model, 
            n_layers=self.n_layers, heads=self.heads, dropout=self.dropout
        )
        
        # Share state to be identical
        new_mlm.load_state_dict(old_mlm.state_dict())
        
        dummy_input = torch.randint(0, self.vocab_size, (2, self.seq_len))
        
        old_mlm.eval()
        new_mlm.eval()
        old_out = old_mlm(dummy_input)
        new_out = new_mlm(dummy_input)
        
        torch.testing.assert_close(old_out, new_out)
        
    def test_mlm_trainer_parity(self):
        torch.manual_seed(42)
        old_bgc_mlm = BGC_MLM_tools.BGC_MLM(
            vocab_size=self.vocab_size, seq_len=self.seq_len, d_model=self.d_model, 
            n_layers=self.n_layers, heads=self.heads, dropout=self.dropout
        )
        old_mlm = BGC_MLM_tools.MLM(old_bgc_mlm, self.vocab_size)
        
        torch.manual_seed(42)
        new_mlm = model_wrappers.create_mlm_model(
            vocab_size=self.vocab_size, seq_len=self.seq_len, d_model=self.d_model, 
            n_layers=self.n_layers, heads=self.heads, dropout=self.dropout
        )
        new_mlm.load_state_dict(old_mlm.state_dict())
        
        # Create identical dataset
        random.seed(42)
        old_ds = BGC_MLM_tools.MLMDataset(self.bgc_tokens, self.token_list, self.seq_len)
        random.seed(42)
        new_ds = data_wrappers.MLMDataset(self.bgc_tokens, self.token_list, self.seq_len)
        
        # Note: test the validation method which computes an aggregate loss using the NLLLoss criterion
        # We use a dummy dataloader by just passing the dataset
        
        # Test Validation computation (does forward pass and criterion computation)
        old_trainer = BGC_MLM_tools.MLMTrainer(
            model=old_mlm,
            train_dataloader=None,
            val_data=None, # Not using during val explicitly
            device='cpu'
        )
        # Using the wrapper equivalent
        new_trainer = model_wrappers.MLMTrainer(
            model=new_mlm,
            train_dataloader=None,
            val_data=None,
            device='cpu' # Using standard trainers
        )
        
        random.seed(42)
        torch.manual_seed(42)
        old_val_loss = old_trainer.validate(old_ds, batch_size=1)
        
        random.seed(42)
        torch.manual_seed(42)
        new_val_loss = new_trainer.validate(new_ds, batch_size=1)
        
        print("OLD VAL LOSS", old_val_loss)
        print("NEW VAL LOSS", new_val_loss)
        
        # Verify dummy pass manually
        old_mlm.eval()
        new_mlm.eval()
        dummy_input = old_ds[0]["bert_input"].unsqueeze(0)
        old_out = old_mlm(dummy_input)
        new_out = new_mlm(dummy_input)
        print("Old MLM Out:", old_out.sum().item(), torch.isnan(old_out).sum())
        print("New MLM Out:", new_out.sum().item(), torch.isnan(new_out).sum())
        
        self.assertAlmostEqual(old_val_loss, new_val_loss, places=4, msg="Val loss mismatch in trainers")

if __name__ == '__main__':
    unittest.main()
