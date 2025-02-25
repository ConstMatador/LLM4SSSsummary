import os
import logging
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler     # pyright:ignore

from utils.conf import Configuration
from utils.sample import getSamples
from utils.loss import ScaledL2Loss
from utils.sample import TSData
from model.GPT4SSS import GPT4SSS


class Experiment:
    def __init__(self, conf:Configuration):
        self.conf = conf
        self.device = self.conf.getEntry("device")
        self.epoch_max = self.conf.getEntry("epoch_max")
        self.model_path = self.conf.getEntry("model_path")
        self.batch_size = self.conf.getEntry("batch_size")
        self.log_path = self.conf.getEntry("log_path")
        
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s',
            filename = self.log_path,
            filemode = "w"
        )
        
    
    def run(self) -> None:
        self.setup()
        
        self.epoch = 0
        self.validate()
        
        while self.epoch < self.epoch_max:
            self.epoch += 1
            
            self.train()
            self.validate()
            
            if self.epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"{self.model_path}example_model.pth")
                logging.info(f"Model in epoch: {self.epoch} saved successfully.")
            
            torch.cuda.empty_cache()

        self.test()
        
    
    def setup(self) -> None:
        
        self.len_series = self.conf.getEntry("len_series")
        self.len_reduce = self.conf.getEntry("len_reduce")
        
        train_sample, val_sample, test_sample = getSamples(self.conf)
        # train_sample, val_sample, test_sample: (train_size, len_series), (val_size, len_series), (test_size, len_series)
        
        self.train_loader1 =  DataLoader(TSData(train_sample), batch_size = self.batch_size, shuffle = True, drop_last=False)
        self.train_loader2 =  DataLoader(TSData(train_sample), batch_size = self.batch_size, shuffle = True, drop_last=False)
        self.val_loader1 = DataLoader(TSData(val_sample), batch_size = self.batch_size, shuffle = True, drop_last=False)
        self.val_loader2 = DataLoader(TSData(val_sample), batch_size = self.batch_size, shuffle = True, drop_last=False)
        self.test_loader1 = DataLoader(TSData(test_sample), batch_size = self.batch_size, shuffle = True, drop_last=False)
        self.test_loader2 = DataLoader(TSData(test_sample), batch_size = self.batch_size, shuffle = True, drop_last=False)
        
        model_selected = self.conf.getEntry("model_selected")
        if model_selected == "GPT4SSS":
            self.model = GPT4SSS(self.conf).to(self.device)
        
        logging.info("Experiment Configuration:")
        for key, value in self.conf.confLoaded.items():
            logging.info(f"{key}: {value}")
        
        if os.path.exists(f"{self.model_path}example_model.pth"):
            logging.info("Model loading...")
            self.model.load_state_dict(torch.load(f"{self.model_path}example_model.pth"))
        else:
            logging.info("Model initializing...")
            self.model = self.InitModel(self.model)
            
        self.loss_calculator = ScaledL2Loss(self.len_series, self.len_reduce)
        
        self.optimizer = AdamW(self.model.parameters(), lr = 1e-4, weight_decay = 0.01)
        self.scaler = GradScaler()
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=False)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda = self.lr_lambda)
        
        # multi-GPUs
        if torch.cuda.device_count() > 1:
            selected_device = self.conf.getEntry("GPUs")
            logging.info(f"Using {len(selected_device)} GPUs")
            self.model = nn.DataParallel(self.model, device_ids=selected_device)
    

    def InitModel(self, model):
        if model is None:
            raise ValueError("The `model` passed to `InitModel` is None.")

        if not isinstance(model, torch.nn.Module):
            raise TypeError("The `model` passed to `InitModel` is not an instance of `torch.nn.Module`.")

        def init_weights(module):
            """
            Initialize weights for specific layer types:
            - nn.Linear: Xavier initialization
            - nn.Conv1d: Kaiming initialization
            - nn.Embedding: Normal distribution initialization
            - nn.LayerNorm: Ones for weights, zeros for biases
            """
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Apply initialization to all submodules except certain ones
        for name, module in model.named_modules():
            if 'llm' in name:
                continue
            init_weights(module)
        return model
            
            
    def train(self) -> None:
        logging.info(f'epoch: {self.epoch}, start training')
        
        for one_batch, another_batch in zip(self.train_loader1, self.train_loader2):
            
            self.optimizer.zero_grad()
            
            one_batch = one_batch.to(self.device)   # (batch_size, len_series)
            one_batch_reduce = self.model(one_batch)   # (batch_size, len_reduce)
            
            with torch.no_grad():
                another_batch = another_batch.to(self.device)
                another_batch_reduce = self.model(another_batch)
                             
            loss = self.loss_calculator(one_batch, another_batch, one_batch_reduce, another_batch_reduce)
            
            self.scaler.scale(loss).backward()      # pyright:ignore
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    
    def validate(self) -> None:
        errors = []
        
        with torch.no_grad():
            for one_batch, another_batch in zip(self.val_loader1, self.val_loader2):
                
                one_batch = one_batch.to(self.device)
                another_batch = another_batch.to(self.device)
                
                one_batch_reduce = self.model(one_batch)
                another_batch_reduce = self.model(another_batch)
                
                err = self.loss_calculator(one_batch, another_batch, one_batch_reduce, another_batch_reduce)
                
                errors.append(err.cpu())
                
        avg_error = torch.mean(torch.stack(errors)).item()
        logging.info(f'epoch: {self.epoch}, validate trans_err: {avg_error:.8f}')
        
        self.scheduler.step(avg_error)
    
    
    def test(self) -> None:
        errors = []

        with torch.no_grad():
            for one_batch, another_batch in zip(self.test_loader1, self.test_loader2):
                one_batch = one_batch.to(self.device)
                another_batch = another_batch.to(self.device)
                one_batch_reduce = self.model(one_batch)
                another_batch_reduce = self.model(another_batch)

                err = self.loss_calculator(one_batch, another_batch, one_batch_reduce, another_batch_reduce)
                errors.append(err.cpu())

        avg_error = torch.mean(torch.stack(errors)).item()
        logging.info(f'test trans_err: {avg_error:.8f}')