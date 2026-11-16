import os
import copy
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import tensorly as tl

from tucker_compressor import (replace_layers_with_tucker, get_model_size_mb,
                               benchmark_latency, count_macs, tucker_layer_to_weight)

tl.set_backend('pytorch')

# =============================================================================
# 1. CONFIGURAÇÃO
# =============================================================================
BATCH_SIZE   = 8
FT_EPOCHS    = 15
FT_LR        = 1e-4
TARGET_FOLD  = 1
SEED         = 42
BASE_DIR     = "../Originais"
HOLDOUT_TEST_SIZE = 0.2   # mesma fração usada no treino original (holdout cego)

THRESHOLDS   = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

NUM_WORKERS  = 4
DETERMINISTIC = False
EARLY_STOP_PATIENCE = 5
BOOTSTRAP_N  = 1000
LATENCY_SHAPE = (1, 3, 224, 224)   # latência por imagem (batch=1)

def extract_patient_id(filepath: str):
    return None

PRETRAINED = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# =============================================================================
# 2. UTILITÁRIOS
# =============================================================================
def set_seed(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = DETERMINISTIC
        torch.backends.cudnn.benchmark = not DETERMINISTIC

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def free_gpu(*objs):
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_model(num_classes):
    weights = models.ResNet50_Weights.DEFAULT if PRETRAINED else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

data_transforms = {
    'treino': transforms.Compose([
        transforms.RandomResizedCrop(size=[224, 224], scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'teste': transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


# =============================================================================
# 3. AVALIAÇÃO / BOOTSTRAP / FINE-TUNING
# =============================================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps, ss = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast('cuda', enabled=USE_AMP):
            out = model(images)
        probs = F.softmax(out.float(), 1)
        ps.extend(torch.argmax(out, 1).cpu().numpy())
        ss.append(probs.cpu().numpy())
        ys.extend(labels.numpy())
    y, p, s = np.array(ys), np.array(ps), np.concatenate(ss, axis=0)
    try:
        auc = roc_auc_score(y, s, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return dict(f1=f1_score(y, p, average='macro'), auc=auc,
                cm=confusion_matrix(y, p), y=y, preds=p, probs=s)

def bootstrap_auc_ci(y, s, n=BOOTSTRAP_N, seed=SEED):
    n_classes = len(np.unique(y))
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); v = []
    for _ in range(n):
        bi = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[bi])) < n_classes:
            continue
        v.append(roc_auc_score(y[bi], s[bi], multi_class='ovr', average='macro'))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float('nan'),)*2

def paired_delta_auc_ci(y, base_s, comp_s, n=BOOTSTRAP_N, seed=SEED):
    n_classes = len(np.unique(y))
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        bi = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[bi])) < n_classes:
            continue
        d.append(roc_auc_score(y[bi], comp_s[bi], multi_class='ovr', average='macro')
                  - roc_auc_score(y[bi], base_s[bi], multi_class='ovr', average='macro'))
    if not d:
        return (float('nan'), float('nan'), False)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return (float(lo), float(hi), bool((lo > 0) or (hi < 0)))

def finetune(model, train_loader, val_loader, class_weights, ckpt_path):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_EPOCHS, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=USE_AMP)
    best, no_improve = -1.0, 0
    for epoch in range(FT_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=USE_AMP):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        scheduler.step()
        res = evaluate(model, val_loader); va = res['auc']
        flag = ""
        if not np.isnan(va) and va > best:
            best, no_improve = va, 0; torch.save(model.state_dict(), ckpt_path); flag = " *"
        else:
            no_improve += 1
        print(f"   FT {epoch+1:02d}/{FT_EPOCHS} | LR {scheduler.get_last_lr()[0]:.6f} "
              f"| Val AUC {va:.4f} | Val F1 {res['f1']:.4f}{flag}")
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"   [early stop]"); break
    return best


# =============================================================================
# 4. ERRO DE RECONSTRUÇÃO POR CAMADA (Tucker)
# =============================================================================
def per_layer_reconstruction_error(original_model, compressed_model):
    snap = {n: m.weight.detach().cpu().clone()
            for n, m in original_model.named_modules()
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)
            and m.groups == 1 and m.in_channels > 64}
    errs = {}
    for name, mod in compressed_model.named_modules():
        if hasattr(mod, 'reduce') and hasattr(mod, 'expand') and name in snap:
            W = tucker_layer_to_weight(mod).cpu()
            w0 = snap[name]
            errs[name] = (torch.linalg.norm(w0 - W) / torch.linalg.norm(w0) * 100).item()
    return errs


# =============================================================================
# 5. SPLIT DE VALIDAÇÃO DO FT
# =============================================================================
def make_ft_split(samples, labels):
    groups = [extract_patient_id(p) for p, _ in samples]
    if all(g is None for g in groups):
        print("[AVISO] Split de validação do FT POR IMAGEM (sem id de paciente).")
        return train_test_split(np.arange(len(labels)), test_size=0.2,
                                random_state=SEED, stratify=labels)
    print("[OK] Split de validação do FT POR PACIENTE.")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return next(sgkf.split(np.zeros(len(labels)), labels, np.array(groups)))


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    set_seed()
    print(f"--- Sweep TUCKER-canal + FT | {device} | AMP={USE_AMP} ---")

    # "Originais" é uma única pasta com 4 classes (healthy/mild/moderate/severe),
    # sem subpastas train/test — refazemos aqui o mesmo holdout cego 80/20 usado
    # no treino original (pipeline_notebook_resnet50_originais.ipynb).
    full_aug   = datasets.ImageFolder(BASE_DIR, data_transforms['treino'])
    full_plain = datasets.ImageFolder(BASE_DIR, data_transforms['teste'])

    num_classes = len(full_aug.classes)
    print(f"[class_to_idx] {full_aug.class_to_idx} ({num_classes} classes)")

    labels = np.array(full_aug.targets)
    all_idx = np.arange(len(labels))
    cv_idx, test_idx = train_test_split(all_idx, test_size=HOLDOUT_TEST_SIZE,
                                        random_state=SEED, stratify=labels)

    cv_samples = [full_aug.samples[i] for i in cv_idx]
    tr_rel, va_rel = make_ft_split(cv_samples, labels[cv_idx])
    tr_idx, va_idx = cv_idx[tr_rel], cv_idx[va_rel]

    persistent = NUM_WORKERS > 0
    ft_train_loader = DataLoader(Subset(full_aug, tr_idx), batch_size=BATCH_SIZE, shuffle=True,
                                 drop_last=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=persistent)
    ft_val_loader   = DataLoader(Subset(full_plain, va_idx), batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=persistent)
    test_loader     = DataLoader(Subset(full_plain, test_idx), batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=persistent)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels[tr_idx]), y=labels[tr_idx])

    ckpt = f"../resnet50_fold_{TARGET_FOLD}_originais_best.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} não encontrado.")
    original = build_model(num_classes)
    original.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    original.eval()
    params_orig = count_parameters(original)
    size_orig = get_model_size_mb(original)
    torch.save(original.state_dict(), "_orig_tmp.pth")
    disk_orig = os.path.getsize("_orig_tmp.pth") / 1024**2; os.remove("_orig_tmp.pth")

    original.to(device)
    base = evaluate(original, test_loader)
    base_lo, base_hi = bootstrap_auc_ci(base['y'], base['probs'])
    macs_orig = count_macs(original, device, LATENCY_SHAPE)
    lat_orig = benchmark_latency(original, device, LATENCY_SHAPE, use_amp=USE_AMP)
    original.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"[BASELINE] fold-{TARGET_FOLD}: F1={base['f1']:.4f} | AUC={base['auc']:.4f} "
          f"[IC95% {base_lo:.4f}–{base_hi:.4f}]")
    print(f"Recursos: {params_orig/1e6:.2f}M params | {disk_orig:.2f} MB disco | "
          f"{macs_orig/1e9:.2f} GMACs | latência {lat_orig['median_ms']:.2f} ms (mediana)")
    print("=" * 70)

    rows = []
    for thr in THRESHOLDS:
        print(f"\n{'#'*60}\n# TUCKER | THRESHOLD = {thr}\n{'#'*60}")
        model = copy.deepcopy(original)
        replace_layers_with_tucker(model, energy_threshold=thr)
        model.to(device)

        params_comp = count_parameters(model)
        size_comp = get_model_size_mb(model)
        ratio = size_orig / size_comp
        macs_comp = count_macs(model, device, LATENCY_SHAPE)
        macs_ratio = macs_orig / macs_comp
        errs = per_layer_reconstruction_error(original, model)
        err_med = float(np.mean(list(errs.values()))) if errs else float('nan')
        print(f"[compressão] params {params_orig/1e6:.2f}M->{params_comp/1e6:.2f}M (ratio {ratio:.2f}x) | "
              f"MACs {macs_orig/1e9:.2f}->{macs_comp/1e9:.2f} G ({macs_ratio:.2f}x menos) | erro recon {err_med:.1f}%")

        if size_comp >= size_orig:
            print(f"[ABORTAR] threshold {thr} inchou (ratio {ratio:.2f}x). Pulando FT.")
            rows.append(dict(threshold=thr, ratio=round(ratio, 3), macs_ratio=round(macs_ratio, 3),
                             recon_err=round(err_med, 2), params_m=round(params_comp/1e6, 2),
                             size_mb=round(size_comp, 2), latency_ms=None, speedup=None,
                             test_f1=None, test_auc=None, auc_lo=None, auc_hi=None,
                             delta_auc=None, d_lo=None, d_hi=None, sig=None, status="inchou"))
            free_gpu(model); continue

        # latência ANTES do FT (estrutura já está fixa; FT não muda shapes/FLOPs)
        lat = benchmark_latency(model, device, LATENCY_SHAPE, use_amp=USE_AMP)
        speedup = lat_orig['median_ms'] / lat['median_ms']

        ft_ckpt = f"resnet50_tucker_e{thr}_ft.pth"
        finetune(model, ft_train_loader, ft_val_loader, class_weights, ft_ckpt)
        model.load_state_dict(torch.load(ft_ckpt, map_location=device, weights_only=True))
        disk_comp = os.path.getsize(ft_ckpt) / 1024**2

        res = evaluate(model, test_loader)
        auc_lo, auc_hi = bootstrap_auc_ci(res['y'], res['probs'])
        d_lo, d_hi, sig = paired_delta_auc_ci(base['y'], base['probs'], res['probs'])
        delta = res['auc'] - base['auc']; cm = res['cm']
        print(f"[teste@{thr}] F1={res['f1']:.4f} | AUC={res['auc']:.4f} [IC95% {auc_lo:.4f}–{auc_hi:.4f}] "
              f"| ΔAUC={delta:+.4f} [{d_lo:+.4f},{d_hi:+.4f}] {'SIG' if sig else 'n.s.'}")
        print(f"   latência {lat['median_ms']:.2f} ms ({speedup:.2f}x mais rápido) | disco {disk_comp:.2f} MB ({disk_orig/disk_comp:.2f}x)")
        print(f"   CM ({', '.join(full_aug.classes)}):\n{cm}")
        rows.append(dict(threshold=thr, ratio=round(disk_orig/disk_comp, 3), macs_ratio=round(macs_ratio, 3),
                         recon_err=round(err_med, 2), params_m=round(params_comp/1e6, 2), size_mb=round(disk_comp, 2),
                         latency_ms=round(lat['median_ms'], 2), speedup=round(speedup, 3),
                         test_f1=round(res['f1'], 4), test_auc=round(res['auc'], 4),
                         auc_lo=round(auc_lo, 4), auc_hi=round(auc_hi, 4),
                         delta_auc=round(delta, 4), d_lo=round(d_lo, 4), d_hi=round(d_hi, 4),
                         sig=sig, status="ok"))
        free_gpu(model)

    # === TABELA DE PARETO ===
    print("\n" + "=" * 110)
    print(f" PARETO TUCKER — baseline AUC={base['auc']:.4f}, {params_orig/1e6:.2f}M params, "
          f"{macs_orig/1e9:.2f} GMACs, {lat_orig['median_ms']:.2f} ms")
    print("=" * 110)
    hdr = (f"{'thr':>6} | {'disco':>5} | {'MACs':>5} | {'lat_ms':>6} | {'speedup':>7} | "
           f"{'recon%':>6} | {'F1':>6} | {'AUC':>6} | {'ΔAUC':>7} | {'sig':>3} | status")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if r['status'] == "ok":
            print(f"{r['threshold']:>6} | {r['ratio']:>5.2f} | {r['macs_ratio']:>5.2f} | "
                  f"{r['latency_ms']:>6.2f} | {r['speedup']:>6.2f}x | {r['recon_err']:>6.1f} | "
                  f"{r['test_f1']:>6.4f} | {r['test_auc']:>6.4f} | {r['delta_auc']:>+7.4f} | "
                  f"{'sim' if r['sig'] else 'não':>3} | ok")
        else:
            print(f"{r['threshold']:>6} | {r['ratio']:>5.2f} | {r['macs_ratio']:>5.2f} | "
                  f"{'-':>6} | {'-':>7} | {r['recon_err']:>6.1f} | {'-':>6} | {'-':>6} | "
                  f"{'-':>7} | {'-':>3} | inchou")

    out_csv = "tucker_sweep_pareto.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["timestamp", "method", "fold", "baseline_f1", "baseline_auc", "baseline_params_m",
                  "baseline_gmacs", "baseline_latency_ms", "threshold", "ratio", "macs_ratio", "recon_err",
                  "params_m", "size_mb", "latency_ms", "speedup", "test_f1", "test_auc", "auc_lo", "auc_hi",
                  "delta_auc", "d_lo", "d_hi", "sig", "status"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            w.writerow(dict(timestamp=ts, method="tucker", fold=TARGET_FOLD,
                            baseline_f1=round(base['f1'], 4), baseline_auc=round(base['auc'], 4),
                            baseline_params_m=round(params_orig/1e6, 2), baseline_gmacs=round(macs_orig/1e9, 3),
                            baseline_latency_ms=round(lat_orig['median_ms'], 2), **r))
    print(f"\n[MLOps] Pareto salvo em: {out_csv}")
    print("\nDiferença-chave vs TT: a coluna 'MACs' e 'speedup' agora SE MOVEM. "
          "No TT elas ficariam em 1.00x para todo threshold (kernel denso reconstruído). "
          "Escolha o maior 'speedup' com ΔAUC 'não' significativo.")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
