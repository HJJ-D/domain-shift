"""Dataset and data loading utilities"""
import os
import re
import random
import json
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


MACHINES = re.compile(r'^\d+$')


def find_all_images(root: Path):
    """Find all PNG images recursively under root directory"""
    files = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() == ".png":
            files.append(p)
    return files


def parse_from_path(path: Path, root: Path):
    """Extract class_name and machine id from path"""
    rel = path.relative_to(root)
    parts = rel.parts
    
    if len(parts) == 0:
        return None
    
    class_name = parts[0]
    machine = None
    for part in parts:
        if MACHINES.match(part):
            machine = part
            break
    
    return class_name, machine


def build_records(root: Path):
    """Build dataset records by scanning directory structure"""
    files = find_all_images(root)
    records = []
    classes = set()
    domains = set()
    excluded_classes = set(['unknown'])
    
    for f in files:
        parsed = parse_from_path(f, root)
        if parsed is None:
            continue
        class_name, domain = parsed
        if class_name is None or class_name.lower() in excluded_classes:
            continue
        
        domains.add(domain if domain is not None else -1)
        classes.add(class_name)
        records.append({
            'path': str(f),
            'class_name': class_name,
            'domain': str(domain) if domain is not None else '-1'
        })
    
    classes_all = sorted(list(classes))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes_all)}
    
    for rec in records:
        rec['label'] = class_to_idx[rec['class_name']]
    
    return records, classes_all, sorted([d for d in domains if d is not None]), class_to_idx


def save_records_to_csv(records, classes_all, domains_all, class_to_idx, csv_path):
    """Save records to CSV file"""
    print(f"Saving {len(records)} records to {csv_path}...")
    
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    
    meta_path = csv_path.replace('.csv', '_meta.json')
    meta = {
        'classes_all': classes_all,
        'domains_all': domains_all,
        'class_to_idx': class_to_idx,
        'num_records': len(records),
        'num_classes': len(classes_all)
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved metadata to {meta_path}")


def load_records_from_csv(csv_path):
    """Load records from CSV cache file"""
    print(f"Loading records from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    df['domain'] = df['domain'].astype(str)
    records = df.to_dict('records')
    
    meta_path = csv_path.replace('.csv', '_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"Loaded {len(records)} records")
    return records, meta['classes_all'], meta['domains_all'], meta['class_to_idx']


def stratified_split(records, test_size=0.2, seed=42):
    """Stratified split by (label, domain) combination"""
    keys = [f"{r['label']}_{r['domain']}" for r in records]
    tr, val = train_test_split(records, test_size=test_size, random_state=seed, stratify=keys)
    return tr, val


def domain_aware_split(records, train_domains=None, val_domains=None, test_domains=None,
                       val_ratio=0.2, test_ratio=0.1, seed=42, domain_sample_ratios=None):
    """Split dataset based on domains for domain generalization"""
    random.seed(seed)
    
    all_domains = sorted(list(set([str(r['domain']) for r in records if r['domain'] != '-1'])))
    
    if len(all_domains) == 0:
        raise ValueError("No valid domains found")
    
    print(f"\n=== Domain-Aware Split ===")
    print(f"Total domains: {len(all_domains)}")
    
    if train_domains is not None:
        train_domains = [str(d) for d in train_domains]
        remaining = [d for d in all_domains if d not in train_domains]
        
        if val_domains is not None and test_domains is not None:
            val_domains = [str(d) for d in val_domains]
            test_domains = [str(d) for d in test_domains]
        elif val_domains is not None:
            val_domains = [str(d) for d in val_domains]
            test_domains = [d for d in remaining if d not in val_domains]
        elif test_domains is not None:
            test_domains = [str(d) for d in test_domains]
            val_domains = [d for d in remaining if d not in test_domains]
        else:
            random.shuffle(remaining)
            n_test = max(1, int(len(remaining) * 0.5))
            test_domains = remaining[:n_test]
            val_domains = remaining[n_test:]
    
    print(f"Train domains: {train_domains}")
    print(f"Val domains: {val_domains}")
    print(f"Test domains: {test_domains}")
    
    train_records = [r for r in records if r['domain'] in train_domains]
    val_records = [r for r in records if r['domain'] in val_domains]
    test_records = [r for r in records if r['domain'] in test_domains]
    
    # Apply domain sampling
    if domain_sample_ratios is not None:
        sampled_train_records = []
        for domain in train_domains:
            domain_records = [r for r in train_records if r['domain'] == domain]
            ratio = domain_sample_ratios.get(domain, 1.0)
            
            if ratio < 1.0:
                n_samples = int(len(domain_records) * ratio)
                random.shuffle(domain_records)
                sampled_records = domain_records[:n_samples]
                print(f"  Domain {domain}: {len(domain_records)} -> {len(sampled_records)}")
                sampled_train_records.extend(sampled_records)
            else:
                sampled_train_records.extend(domain_records)
        
        train_records = sampled_train_records
    
    unknown_records = [r for r in records if r['domain'] == '-1']
    if unknown_records:
        print(f"Adding {len(unknown_records)} unknown domain records to training")
        train_records.extend(unknown_records)
    
    print(f"Final: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    
    return train_records, val_records, test_records


class RiceDataset(Dataset):
    """Rice grain dataset"""
    
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['path']).convert('RGB')
        except Exception:
            new_idx = random.randint(0, len(self.records) - 1)
            return self.__getitem__(new_idx)
        
        if self.transform:
            img = self.transform(img)
        
        label = rec['label']
        domain_id = rec.get('domain_id', -1)
        
        return img, torch.tensor(label, dtype=torch.long), torch.tensor(domain_id, dtype=torch.long)