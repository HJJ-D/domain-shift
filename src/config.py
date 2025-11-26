"""Configuration management for domain shift training"""
import json
from pathlib import Path
from datetime import datetime


class Config:
    """Training configuration class"""
    
    def __init__(self, base_path='/proj/uppmax2025-2-369/Cgrain'):
        self.base_path = base_path
        self.input_path = Path(f'{base_path}/segmented')
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'./runs/run_multi_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data parameters
        self.img_size = 224
        self.batch_size = 64
        self.num_workers = 8
        
        # Training parameters
        self.epochs = 20
        self.lr = 1e-4
        self.lr_scheduler = 'cosine'
        self.warmup_epochs = 2
        self.optimizer = 'adamw'
        self.weight_decay = 1e-4
        self.max_grad_norm = 1.0
        
        # DANN parameters
        self.use_dann = True
        self.lambda_dann = 0.5
        
        # Domain split parameters
        self.split_strategy = 'domain_aware'
        self.train_domains = ['848', '899', '900', '895']
        self.val_domains = ['901', '898', '896']
        self.test_domains = ['871', '882', '890']
        self.domain_sample_ratios = {
            '848': 0.33,
            '899': 0.33,
            '900': 0.33,
            '895': 1.0
        }
        self.val_ratio = 0.10
        self.test_ratio = 0.10
        
        # Single machine parameters
        self.train_per_machine = False
        self.target_machine = '900'
        
        # Other parameters
        self.seed = 42
        self.csv_cache_path = f'{base_path}/dataset_records_multi.csv'
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'root': str(self.input_path),
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'epochs': self.epochs,
            'lr': self.lr,
            'lr_scheduler': self.lr_scheduler,
            'warmup_epochs': self.warmup_epochs,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'use_dann': self.use_dann,
            'lambda_dann': self.lambda_dann,
            'split_strategy': self.split_strategy,
            'train_domains': self.train_domains,
            'val_domains': self.val_domains,
            'test_domains': self.test_domains,
            'domain_sample_ratios': self.domain_sample_ratios,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'train_per_machine': self.train_per_machine,
            'target_machine': self.target_machine,
            'output_dir': str(self.output_dir),
            'seed': self.seed
        }
    
    def save(self):
        """Save config to JSON file"""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved config to {config_path}")