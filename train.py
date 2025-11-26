"""Main training script for domain shift rice grain classification"""
import os
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import classification_report
from PIL import Image

from src.config import Config
from src.dataset import (
    build_records, load_records_from_csv, save_records_to_csv,
    domain_aware_split, stratified_split, RiceDataset
)
from src.models import build_backbone, ClassifierHead, DomainClassifier
from src.trainer import train_one_epoch, evaluate, evaluate_domain
from src.utils import (
    set_seed, setup_logging, log_print, save_checkpoint,
    create_domain_mapping, save_domain_mapping, save_split_info,
    save_training_history
)


def main():
    # Initialize configuration
    config = Config()
    config.save()
    
    print(f"Output directory: {config.output_dir}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logging
    log_file = setup_logging(config.output_dir)
    
    log_print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"Config: {json.dumps(config.to_dict(), indent=2)}", log_file)
    
    # ---------- Build or load dataset records ----------
    if os.path.exists(config.csv_cache_path):
        log_print(f"Loading records from cache: {config.csv_cache_path}", log_file)
        records, classes_all, domains_all, class_to_idx = load_records_from_csv(config.csv_cache_path)
    else:
        log_print("Building records from scratch...", log_file)
        records, classes_all, domains_all, class_to_idx = build_records(config.input_path)
        save_records_to_csv(records, classes_all, domains_all, class_to_idx, config.csv_cache_path)
    
    num_classes = len(classes_all)
    log_print(f"Found {len(records)} images, {num_classes} classes, {len(domains_all)} domains", log_file)
    log_print(f"Classes: {classes_all}", log_file)
    log_print(f"Domains: {domains_all}", log_file)
    
    # ---------- Data splitting ----------
    if config.train_per_machine:
        log_print(f"Training on single machine: {config.target_machine}", log_file)
        records = [r for r in records if r['domain'] == config.target_machine]
        train_records, val_records = stratified_split(records, test_size=0.2, seed=config.seed)
        test_records = []
    else:
        if config.split_strategy == 'domain_aware':
            train_records, val_records, test_records = domain_aware_split(
                records,
                train_domains=config.train_domains,
                val_domains=config.val_domains,
                test_domains=config.test_domains,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                seed=config.seed,
                domain_sample_ratios=config.domain_sample_ratios
            )
        else:
            train_records, val_records = stratified_split(records, test_size=0.2, seed=config.seed)
            test_records = []
    
    log_print(f"Split: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}", log_file)
    
    # ---------- Create domain mapping ----------
    domain_to_id = create_domain_mapping(records)
    for rec in train_records:
        rec['domain_id'] = domain_to_id[rec['domain']]
    for rec in val_records:
        rec['domain_id'] = domain_to_id[rec['domain']]
    for rec in test_records:
        rec['domain_id'] = domain_to_id[rec['domain']]
    
    num_domains = save_domain_mapping(domain_to_id, config.output_dir)
    
    # Save class mapping
    with open(config.output_dir / 'class_mapping.json', 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {v: k for k, v in class_to_idx.items()},
            'num_classes': num_classes
        }, f, indent=2)
    
    # Save split info
    save_split_info(train_records, val_records, test_records, config.output_dir)
    
    # ---------- Data transforms ----------
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ---------- Create datasets and dataloaders ----------
    train_ds = RiceDataset(train_records, transform=train_transform)
    val_ds = RiceDataset(val_records, transform=val_transform)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # ---------- Build models ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Device: {device}", log_file)
    
    backbone, feat_dim = build_backbone('resnet50', pretrained=True)
    backbone = backbone.to(device)
    
    clf_head = ClassifierHead(in_features=feat_dim, num_classes=num_classes).to(device)
    
    domain_clf = None
    if config.use_dann:
        domain_clf = DomainClassifier(in_features=feat_dim, num_domains=num_domains).to(device)
        log_print(f"Created DomainClassifier with {num_domains} domains", log_file)
    
    # ---------- Optimizer and scheduler ----------
    params = list(backbone.parameters()) + list(clf_head.parameters())
    if domain_clf is not None:
        params += list(domain_clf.parameters())
    
    optimizer = torch.optim.AdamW(
        params,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=config.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_epochs]
    )
    
    # ---------- Training loop ----------
    best_f1 = -1
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    log_print("=" * 60, log_file)
    log_print("Starting training...", log_file)
    log_print("=" * 60, log_file)
    
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        
        # Train
        loss_train, acc_train, acc_dom_train = train_one_epoch(
            backbone, clf_head, domain_clf, train_loader, optimizer, device,
            epoch, config.epochs, config.use_dann, config.lambda_dann,
            num_domains, config.max_grad_norm
        )
        
        # Validate
        acc_val, f1_val, cm = evaluate(backbone, clf_head, val_loader, device)
        
        # Evaluate domain classifier on train set
        acc_dom_val = 0.0
        cm_dom = None
        if config.use_dann and domain_clf is not None:
            dom_metrics = evaluate_domain(backbone, domain_clf, train_loader, device)
            if dom_metrics:
                acc_dom_val, cm_dom = dom_metrics
        
        dt = time.time() - t0
        
        # Log results
        if config.use_dann and domain_clf is not None:
            log_msg = (f"Epoch {epoch}/{config.epochs}  "
                      f"train_loss={loss_train:.4f} train_acc={acc_train:.4f} "
                      f"acc_dom_train={acc_dom_train:.4f} "
                      f"val_acc={acc_val:.4f} val_f1={f1_val:.4f} "
                      f"dom_train_eval={acc_dom_val:.4f}  time={dt:.1f}s")
        else:
            log_msg = (f"Epoch {epoch}/{config.epochs}  "
                      f"train_loss={loss_train:.4f} train_acc={acc_train:.4f} "
                      f"val_acc={acc_val:.4f} val_f1={f1_val:.4f}  time={dt:.1f}s")
        
        log_print(log_msg, log_file)
        log_print(f'Confusion matrix:\n{cm}', log_file)
        
        if config.use_dann and cm_dom is not None:
            log_print(f'Domain confusion matrix:\n{cm_dom}', log_file)
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(float(loss_train))
        history['train_acc'].append(float(acc_train))
        history['val_acc'].append(float(acc_val))
        history['val_f1'].append(float(f1_val))
        
        # Save best model
        if f1_val > best_f1:
            best_f1 = f1_val
            best_path = config.output_dir / 'best_model.pth'
            save_checkpoint(
                backbone, clf_head, domain_clf, optimizer, epoch,
                config.to_dict(), classes_all, class_to_idx, domains_all,
                best_path, best_f1
            )
            log_print(f"Saved best model (F1={best_f1:.4f})", log_file)
        
        # Step scheduler
        scheduler.step()
    
    # ---------- Save final model ----------
    final_path = config.output_dir / 'final_model.pth'
    save_checkpoint(
        backbone, clf_head, domain_clf, optimizer, config.epochs,
        config.to_dict(), classes_all, class_to_idx, domains_all, final_path
    )
    log_print(f"Saved final model", log_file)
    
    # Save training history
    save_training_history(history, config.output_dir)
    
    log_print(f"Training complete. Best F1: {best_f1:.4f}", log_file)
    
    # ---------- Test set evaluation ----------
    if len(test_records) > 0:
        log_print("\n" + "=" * 60, log_file)
        log_print("Evaluating on test set (unseen domains)...", log_file)
        log_print("=" * 60, log_file)
        
        test_ds = RiceDataset(test_records, transform=val_transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Load best model
        checkpoint = torch.load(config.output_dir / 'best_model.pth')
        backbone.load_state_dict(checkpoint['backbone_state'])
        clf_head.load_state_dict(checkpoint['clf_state'])
        
        acc_test, f1_test, cm_test = evaluate(backbone, clf_head, test_loader, device)
        log_print(f"Test Accuracy: {acc_test:.4f}", log_file)
        log_print(f"Test F1: {f1_test:.4f}", log_file)
        log_print(f"Test Confusion Matrix:\n{cm_test}", log_file)
        
        # Detailed classification report
        backbone.eval()
        clf_head.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for imgs, labels, _ in test_loader:
                imgs = imgs.to(device)
                logits = clf_head(backbone(imgs))
                preds = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(labels.numpy().tolist())
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        target_names = [idx_to_class[i] for i in range(num_classes)]
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        
        log_print("\nPer-class test performance:", log_file)
        log_print(report, log_file)
        
        # Save report
        with open(config.output_dir / 'test_classification_report.txt', 'w') as f:
            f.write(report)
    
    log_print(f"\nTraining finished at {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"All outputs saved to: {config.output_dir}", log_file)
    log_file.close()
    
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Output directory: {config.output_dir}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"{'=' * 60}")

