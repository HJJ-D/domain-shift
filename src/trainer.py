"""Training and evaluation functions"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .models import grad_reverse


def train_one_epoch(model_backbone, clf_head, domain_clf, loader, optimizer, 
                    device, epoch, max_epochs, use_dann=False, lambda_dann=0.1, 
                    num_domains=10, max_grad_norm=1.0):
    """Train for one epoch"""
    model_backbone.train()
    clf_head.train()
    if domain_clf is not None:
        domain_clf.train()
    
    losses = []
    accs = []
    dom_accs = []
    
    for batch_idx, (imgs, labels, domains) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        
        optimizer.zero_grad()
        
        features = model_backbone(imgs)
        logits = clf_head(features)
        loss_task = F.cross_entropy(logits, labels)
        
        loss = loss_task
        dom_acc = 0.0
        
        if use_dann and domain_clf is not None:
            p = float(epoch) / float(max_epochs)
            lambd = 2. / (1. + np.exp(-10 * p)) - 1
            rev_feat = grad_reverse(features, lambd=lambd * lambda_dann)
            domain_logits = domain_clf(rev_feat)
            
            valid_mask = (domains >= 0)
            if valid_mask.any():
                valid_dom = domains[valid_mask]
                dom_pred = domain_logits[valid_mask]
                dom_acc = (dom_pred.argmax(dim=1) == valid_dom).float().mean().item()
                loss_dom = F.cross_entropy(dom_pred, valid_dom)
                loss = loss + loss_dom
        
        loss.backward()
        
        params = list(model_backbone.parameters()) + list(clf_head.parameters())
        if domain_clf is not None:
            params += list(domain_clf.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        
        optimizer.step()
        
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        accs.append(float((preds == labels.detach().cpu().numpy()).mean()))
        dom_accs.append(dom_acc)
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(loader)} loss={losses[-1]:.4f} "
                  f"acc={accs[-1]:.4f} dom_acc={dom_acc:.4f}")
    
    return np.mean(losses), np.mean(accs), np.mean(dom_accs)


@torch.no_grad()
def evaluate(model_backbone, clf_head, loader, device):
    """Evaluate classification performance"""
    model_backbone.eval()
    clf_head.eval()
    
    y_true = []
    y_pred = []
    
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        logits = clf_head(model_backbone(imgs))
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return acc, f1, cm


@torch.no_grad()
def evaluate_domain(model_backbone, domain_clf, loader, device):
    """Evaluate domain classifier performance"""
    if domain_clf is None:
        return None
    
    model_backbone.eval()
    domain_clf.eval()
    
    dom_true, dom_pred = [], []
    
    for imgs, _, domains in loader:
        imgs = imgs.to(device)
        domains = domains.to(device)
        
        features = model_backbone(imgs)
        domain_logits = domain_clf(features)
        preds = domain_logits.argmax(dim=1)
        
        valid_mask = (domains >= 0)
        dom_true.extend(domains[valid_mask].cpu().numpy())
        dom_pred.extend(preds[valid_mask].cpu().numpy())
    
    if len(dom_true) == 0:
        return None
    
    dom_acc = accuracy_score(dom_true, dom_pred)
    cm_dom = confusion_matrix(dom_true, dom_pred)
    
    return dom_acc, cm_dom
