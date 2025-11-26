import os
from pathlib import Path
import re
import json
import random
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import torch

MACHINES = re.compile(r'^\d+$')

def find_all_images(root: Path):
    files = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            files.append(p)
    return files

def parse_from_path(path: Path, root: Path):
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) == 0:
        return None, None
    class_name = parts[0]
    machine = None
    for part in parts:
        if MACHINES.match(part):
            machine = part
            break
    return class_name, machine

def build_records(root: Path):
    """
    Scan directory, build records and map class names -> labels (multi-class)
    Returns: records, classes_all, domains_all
    records: list of dict with keys: path, class_name, domain (string or '-1'), label (int)
    """
    files = find_all_images(root)
    classes = set()
    domains = set()
    parsed = []
    for f in files:
        cls, dom = parse_from_path(f, root)
        if cls is None:
            continue
        classes.add(cls)
        domains.add(dom if dom is not None else '-1')
        parsed.append((f, cls, dom if dom is not None else '-1'))
    classes_all = sorted(list(classes))
    domains_all = sorted([d for d in domains if d != '-1'])
    class_to_id = {c: i for i, c in enumerate(classes_all)}
    records = []
    for f, cls, dom in parsed:
        records.append({
            'path': str(f),
            'class_name': cls,
            'domain': dom,
            'label': class_to_id[cls]
        })
    return records, classes_all, domains_all

def save_records_to_csv(records, classes_all, domains_all, csv_path):
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    meta = {'classes_all': classes_all, 'domains_all': domains_all, 'num_records': len(records)}
    meta_path = csv_path.replace('.csv', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

def load_records_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    records = df.to_dict('records')
    meta_path = csv_path.replace('.csv', '_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return records, meta['classes_all'], meta['domains_all']

def compute_domain_mapping(records):
    all_domain_strings = sorted(list(set([r['domain'] for r in records if r['domain'] != '-1'])))
    domain_to_id = {domain_str: idx for idx, domain_str in enumerate(all_domain_strings)}
    domain_to_id['-1'] = -1
    id_to_domain = {idx: domain_str for domain_str, idx in domain_to_id.items() if idx != -1}
    return domain_to_id, id_to_domain

def stratified_split(records, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    keys = [f"{r['label']}_{r['domain']}" for r in records]
    tr, val = train_test_split(records, test_size=test_size, random_state=seed, stratify=keys)
    return tr, val

def domain_aware_split(records, val_domains=None, test_domains=None, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    all_domains = sorted(list(set([r['domain'] for r in records if r['domain'] != '-1'])))
    if len(all_domains) == 0:
        raise ValueError("No valid domains found")
    if val_domains is None and test_domains is None:
        shuffled = all_domains.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test_domains = shuffled[:n_test]
        val_domains = shuffled[n_test:n_test + n_val]
        train_domains = shuffled[n_test + n_val:]
    else:
        train_domains = [d for d in all_domains if d not in (val_domains or []) and d not in (test_domains or [])]
    train_records = [r for r in records if r['domain'] in train_domains]
    val_records = [r for r in records if r['domain'] in val_domains]
    test_records = [r for r in records if r['domain'] in test_domains]
    unknown = [r for r in records if r['domain'] == '-1']
    if unknown:
        train_records.extend(unknown)
    return train_records, val_records, test_records

def compute_domain_weights(records, num_domains):
    from collections import Counter
    domain_counts = Counter([r['domain_id'] for r in records if r['domain_id'] >= 0])
    total = sum(domain_counts.values())
    weights = []
    for i in range(num_domains):
        cnt = max(1, domain_counts.get(i, 0))
        weights.append(total / (cnt * num_domains))
    return np.array(weights, dtype=np.float32)

def create_balanced_sampler(records, num_domains):
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter
    domain_counts = Counter([r['domain_id'] for r in records if r['domain_id'] >= 0])
    total = len([r for r in records if r['domain_id'] >= 0])
    sample_weights = []
    for r in records:
        did = r.get('domain_id', -1)
        if did >= 0:
            w = total / (domain_counts[did] * num_domains)
            sample_weights.append(w)
        else:
            sample_weights.append(0.0)
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float32),
                                    num_samples=len(records), replacement=True)
    return sampler

"""Utility functions for logging, checkpointing, and visualization"""

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir):
    """Setup logging to file"""
    log_file = output_dir / 'training.log'
    log_f = open(log_file, 'w')
    return log_f


def log_print(msg, log_file=None):
    """Print and write to log file"""
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()


def save_checkpoint(backbone, clf_head, domain_clf, optimizer, epoch, config, 
                    classes_all, class_to_idx, domains_all, path, best_f1=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'backbone_state': backbone.state_dict(),
        'clf_state': clf_head.state_dict(),
        'domain_state': domain_clf.state_dict() if domain_clf is not None else None,
        'optimizer_state': optimizer.state_dict(),
        'config': config,
        'classes_all': classes_all,
        'class_to_idx': class_to_idx,
        'domains_all': domains_all
    }
    if best_f1 is not None:
        checkpoint['best_f1'] = best_f1
    
    torch.save(checkpoint, path)


def load_checkpoint(path, backbone, clf_head, domain_clf=None, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    backbone.load_state_dict(checkpoint['backbone_state'])
    clf_head.load_state_dict(checkpoint['clf_state'])
    
    if domain_clf is not None and checkpoint['domain_state'] is not None:
        domain_clf.load_state_dict(checkpoint['domain_state'])
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return checkpoint


def save_domain_mapping(domain_to_id, output_dir):
    """Save domain mapping to JSON"""
    id_to_domain = {idx: domain_str for domain_str, idx in domain_to_id.items()}
    num_domains = len([d for d in domain_to_id.keys() if d != '-1'])
    
    mapping = {
        'domain_to_id': domain_to_id,
        'id_to_domain': id_to_domain,
        'num_domains': num_domains
    }
    
    path = output_dir / 'domain_mapping.json'
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Saved domain mapping to {path}")
    return num_domains


def save_split_info(train_records, val_records, test_records, output_dir):
    """Save dataset split information"""
    split_info = {
        'train_size': len(train_records),
        'val_size': len(val_records),
        'test_size': len(test_records),
        'train_domains': sorted(list(set([r['domain'] for r in train_records if r['domain'] != '-1']))),
        'val_domains': sorted(list(set([r['domain'] for r in val_records if r['domain'] != '-1']))),
        'test_domains': sorted(list(set([r['domain'] for r in test_records if r['domain'] != '-1']))),
        'train_domain_ids': sorted(list(set([r['domain_id'] for r in train_records if r['domain_id'] != -1]))),
        'val_domain_ids': sorted(list(set([r['domain_id'] for r in val_records if r['domain_id'] != -1]))),
        'test_domain_ids': sorted(list(set([r['domain_id'] for r in test_records if r['domain_id'] != -1])))
    }
    
    path = output_dir / 'split_info.json'
    with open(path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Saved split info to {path}")


def create_domain_mapping(records):
    """Create domain to ID mapping"""
    all_domain_strings = sorted(list(set([r['domain'] for r in records if r['domain'] != '-1'])))
    domain_to_id = {domain_str: idx for idx, domain_str in enumerate(all_domain_strings)}
    domain_to_id['-1'] = -1
    
    # Apply mapping to records
    for record in records:
        record['domain_id'] = domain_to_id[record['domain']]
    
    return domain_to_id


def save_training_history(history, output_dir):
    """Save training history to JSON"""
    path = output_dir / 'training_history.json'
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {path}")