import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler     # pyright:ignore

from utils.conf import Configuration
from utils.sample import getSamples
from utils.loss import ScaledL2Loss
from utils.sample import TSData

from model.GPT4SSS import GPT4SSS
from model.TimeLLM import TimeLLM
from model.AutoTimes import AutoTimes
from model.UniTime import UniTime
from model.S2IPLLM import S2IPLLM


class Experiment:
    def __init__(self, conf:Configuration):
        self.conf = conf
        self.device = self.conf.getEntry("device")
        self.epoch_max = self.conf.getEntry("epoch_max")
        self.model_path = self.conf.getEntry("model_path")
        self.batch_size = self.conf.getEntry("batch_size")
        self.log_path = self.conf.getEntry("log_path")
        self.model_selected = self.conf.getEntry("model_selected")
        if self.model_selected == "UniTime":
            self.mask_rate = self.conf.getEntry("mask_rate")
            
        # Early Stopping
        self.patience = 10
        self.best_val_error = float('inf')
        self.epochs_without_improvement = 0
        
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
                torch.save(self.model.module.state_dict(), f"{self.model_path}example_model.pth")
                logging.info(f"Model in epoch: {self.epoch} saved successfully.")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logging.info(f"Early stopping at epoch {self.epoch} due to no improvement.")
                break
            
        self.test()
        
    
    def setup(self) -> None:
        
        self.len_series = self.conf.getEntry("len_series")
        self.len_reduce = self.conf.getEntry("len_reduce")
        
        train_sample, val_sample, test_sample = getSamples(self.conf)
        # train_sample, val_sample, test_sample: (train_size, len_series), (val_size, len_series), (test_size, len_series)
        
        self.train_loader1 =  DataLoader(TSData(train_sample), 
                                         batch_size = self.batch_size, shuffle = True, drop_last=True)
        self.train_loader2 =  DataLoader(TSData(train_sample), 
                                         batch_size = self.batch_size, shuffle = True, drop_last=True)
        self.val_loader1 = DataLoader(TSData(val_sample), 
                                      batch_size = self.batch_size, shuffle = True, drop_last=True)
        self.val_loader2 = DataLoader(TSData(val_sample), 
                                      batch_size = self.batch_size, shuffle = True, drop_last=True)
        self.test_loader1 = DataLoader(TSData(test_sample), 
                                       batch_size = self.batch_size, shuffle = True, drop_last=True)
        self.test_loader2 = DataLoader(TSData(test_sample), 
                                       batch_size = self.batch_size, shuffle = True, drop_last=True)
        
        model_selected = self.conf.getEntry("model_selected")
        if model_selected == "GPT4SSS":
            self.model = GPT4SSS(self.conf).to(self.device)
        elif model_selected == "TimeLLM":
            self.model = TimeLLM(self.conf).to(self.device)
        elif model_selected == "AutoTimes":
            self.model = AutoTimes(self.conf).to(self.device)
        elif model_selected == "UniTime":
            self.model = UniTime(self.conf).to(self.device)
        elif model_selected == "S2IPLLM":
            self.model = S2IPLLM(self.conf).to(self.device)
        
        logging.info("Experiment Configuration:")
        for key, value in self.conf.confLoaded.items():
            logging.info(f"{key}: {value}")
        
        if os.path.exists(f"{self.model_path}example_model.pth"):
            logging.info("Model loading...")
            checkpoint = torch.load(f"{self.model_path}example_model.pth")
            # checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
            self.model.load_state_dict(checkpoint)
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
    
    # For UniTime
    def random_mask(self):
        mask = torch.rand((self.batch_size, self.len_series))   # 随机生成0-1的随机数
        # mask: (batch_size, len_series)
        mask[mask < self.mask_rate] = 0  # masked
        mask[mask >= self.mask_rate] = 1  # remained
        # mask: (batch_size, len_series)
        return mask
        
            
    def train(self) -> None:
        logging.info(f'epoch: {self.epoch}, start training')
        
        for one_batch, another_batch in zip(self.train_loader1, self.train_loader2):
            # print(one_batch.shape, another_batch.shape)
            self.optimizer.zero_grad()
            
            # masking
            if self.model_selected == "UniTime":
                one_mask = self.random_mask().to(self.device)
                another_mask = self.random_mask().to(self.device)
                
                one_batch = one_batch.to(self.device)
                one_batch = one_batch.masked_fill(one_mask==0, 0)   # (batch_size, len_series)
                one_batch_reduce = self.model(one_batch, one_mask)  # (batch_size, len_reduce)
                
                with torch.no_grad():
                    another_batch = another_batch.to(self.device)
                    another_batch = another_batch.masked_fill(another_mask==0, 0)
                    another_batch_reduce = self.model(another_batch, another_mask)
            else:
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
                
                # masking
                mask = torch.ones((self.batch_size, self.len_series)).to(self.device)
                with torch.no_grad():
                    if self.model_selected == "UniTime":
                        one_batch_reduce = self.model(one_batch, mask)
                        another_batch_reduce = self.model(another_batch, mask)
                    else:
                        one_batch_reduce = self.model(one_batch)
                        another_batch_reduce = self.model(another_batch)
                        
                err = self.loss_calculator(one_batch, another_batch, one_batch_reduce, another_batch_reduce)
                
                errors.append(err.cpu())
                
        avg_error = torch.mean(torch.stack(errors)).item()
        logging.info(f'epoch: {self.epoch}, validate trans_err: {avg_error:.6f}')
        
        if avg_error < self.best_val_error:
            self.best_val_error = avg_error
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        self.scheduler.step(avg_error)
    
    
    def test(self) -> None:
        errors = []

        with torch.no_grad():
            for one_batch, another_batch in zip(self.test_loader1, self.test_loader2):
                one_batch = one_batch.to(self.device)
                another_batch = another_batch.to(self.device)
                
                # masking
                mask = torch.ones((self.batch_size, self.len_series)).to(self.device)
                with torch.no_grad():
                    if self.model_selected == "UniTime":
                        one_batch_reduce = self.model(one_batch, mask)
                        another_batch_reduce = self.model(another_batch, mask)
                    else:
                        one_batch_reduce = self.model(one_batch)
                        another_batch_reduce = self.model(another_batch)

                err = self.loss_calculator(one_batch, another_batch, one_batch_reduce, another_batch_reduce)
                errors.append(err.cpu())

        avg_error = torch.mean(torch.stack(errors)).item()
        logging.info(f'test trans_err: {avg_error:.6f}')