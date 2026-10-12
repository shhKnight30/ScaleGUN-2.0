from torch.utils.data import Dataset
import torch
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
import math
from numpy.linalg import norm
from sklearn import preprocessing
from torch import Tensor
from pathlib import Path
from torch.nn import init
import logging
import time
from torch.autograd import grad
import gc
import csv

# ═══════════════════════════════════════════════════════
# NEW: sklearn imports for research metrics
# ═══════════════════════════════════════════════════════
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score

logger = logging.getLogger(__name__)


def setup_unlearn_logger(name):
    global logger
    logger = logging.getLogger(name)


# ═══════════════════════════════════════════════════════
# Consistent PGD Projection (used in BOTH training and unlearning)
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def apply_pgd_projection(model, pgd_c, mode='per_param'):
    """
    PGD projection with selectable mode.
    
    'per_param': Each parameter tensor independently bounded (original behavior)
    'global': All parameters jointly bounded
    """
    if mode == 'per_param':
        for p in model.parameters():
            if p.requires_grad:
                p_norm = p.data.norm(2)
                if p_norm > pgd_c:
                    p.data.mul_(pgd_c / (p_norm + 1e-8))
    
    elif mode == 'global':
        total_norm_sq = sum(
            p.data.norm(2).item() ** 2
            for p in model.parameters() if p.requires_grad
        )
        total_norm = total_norm_sq ** 0.5
        if total_norm > pgd_c:
            scale = pgd_c / (total_norm + 1e-8)
            for p in model.parameters():
                if p.requires_grad:
                    p.data.mul_(scale)


# ═══════════════════════════════════════════════════════
# Fixed Training Loop
# ═══════════════════════════════════════════════════════
def train_model(model, device, X_train, y_train, batch_size,
                optimizer, epochs, X_val, y_val, evaluator,
                checkpt_file, patience, verbose=False,
                noises=None, pgd_c=50.0):
    bad_counter = 0
    best = 0
    num_train = X_train.shape[0]
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        loss_list = []
        shuffled_indices = torch.randperm(num_train)
        idx = 0

        while idx < num_train:
            batch_idx = shuffled_indices[idx:idx + batch_size]
            if len(batch_idx) < 2:
                idx += batch_size
                continue
            x = X_train[batch_idx].to(device)
            y = y_train[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.nll_loss(out, y)

            # ── FIX: Use param.data (original behavior, no gradient through noise) ──
            if noises is not None:
                noise_loss = 0
                for ni, param in enumerate(
                    p for p in model.parameters() if p.requires_grad
                ):
                    noise_loss += (param * noises[ni]).sum()
                loss = loss + noise_loss / (num_train * 2)

            loss.backward()
            optimizer.step()

            # ── FIX: Per-parameter PGD (same as original, matches unlearning) ──
            if pgd_c is not None:
                apply_pgd_projection(model, pgd_c, mode='per_param')

            loss_list.append(loss.item())
            idx += batch_size
            del x, y, out

        f1_val = test(model, device, X_val, y_val, batch_size, evaluator)

        if verbose and (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch:{epoch+1:03d} | Loss:{np.mean(loss_list):.4f} | "
                f"Val:{100*f1_val:.2f}%"
            )

        if f1_val > best:
            best = f1_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= patience:
            logger.info(f"Early stop at epoch {epoch}")
            break

    return time.time() - start_time


# ═══════════════════════════════════════════════════════
# Fixed Test Function (counts correct/total, not averaging percentages)
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def test(model, device, X_val, y_val, batch_size, evaluator=None):
    """Correct accuracy: accumulates counts, not percentages."""
    model.eval()
    correct = 0
    total = 0
    idx = 0
    while idx < X_val.shape[0]:
        x = X_val[idx:idx + batch_size].to(device)
        y = y_val[idx:idx + batch_size].to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        idx += batch_size
        del x, y
    return correct / total


# ═══════════════════════════════════════════════════════
# Fixed Gradient Computation (memory-efficient)
# ═══════════════════════════════════════════════════════
def cal_grad_handloader(model, device, _X, _y, batch_size, retain=False):
    """
    Memory-efficient gradient computation.
    When retain=False: uses standard backward accumulation (fast, low memory).
    When retain=True: accumulates per-batch gradients with graph (needed for HVP).
    """
    model.eval()
    model.zero_grad()
    params = [p for p in model.parameters() if p.requires_grad]
    num_data = _X.shape[0]

    if not retain:
        # Fast path: standard backward
        for start in range(0, num_data, batch_size):
            x = _X[start:start + batch_size].to(device)
            y = _y[start:start + batch_size].to(device)
            
            if x.shape[0] < 2:
               continue
            
            out = model(x)
            loss = F.nll_loss(out, y, reduction='sum') / num_data
            loss.backward()
            del x, y, out, loss
        grads = tuple(p.grad.clone() for p in params)
        model.zero_grad()
        return grads
    else:
        # Retain path: per-batch grad accumulation with graph
        accumulated = [torch.zeros_like(p) for p in params]
        for start in range(0, num_data, batch_size):
            x = _X[start:start + batch_size].to(device)
            y = _y[start:start + batch_size].to(device)
            
            if x.shape[0] < 2:
               continue
            
            out = model(x)
            loss = F.nll_loss(out, y, reduction='sum') / num_data
            batch_grads = grad(loss, params, create_graph=True)
            for j in range(len(accumulated)):
                accumulated[j] = accumulated[j] + batch_grads[j]
            del x, y, out, loss, batch_grads
        return tuple(accumulated)


# ═══════════════════════════════════════════════════════
# HVP helpers (unchanged — these are correct)
# ═══════════════════════════════════════════════════════
def hvps(grad_all, model_params, h_estimate):
    element_product = torch.tensor(0.0, device=grad_all[0].device)
    for grad_elem, v_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * v_elem)
    return_grads = grad(element_product, model_params, create_graph=True)
    return return_grads


# ═══════════════════════════════════════════════════════
# Fixed LiSSA
# ═══════════════════════════════════════════════════════
def lissa_inverse_hvp(v, model, device, X_train, y_train,
                       batch_size=256, recursions=500,
                       damp=0.01, scale=10.0,
                       convergence_tol=1e-4, check_every=25):
    """
    LiSSA inverse-Hessian-vector product.

    Implements the recursion from Agarwal et al. (2017) and CertDNN
    (Zhang et al., ICML 2024, Algorithm 1):

        H̃⁻¹_t = I + (I − H_t / scale) × H̃⁻¹_{t−1}

    The output is H̃⁻¹_s / scale  (one division by scale total).

    FIX vs previous version: the old code divided h_estimate by scale
    BOTH inside the loop (via `... / scale`) AND again at the end
    (`h_estimate = [h / scale ...]`), making the result (H̃⁻¹v) / scale²
    instead of (H̃⁻¹v) / scale.  The final explicit division is now removed.

    Reference: CertDNN Appendix D — s = 1000 recursions (no early stopping).
    With convergence_tol early stopping, 500 max iterations is sufficient.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    num_train = X_train.shape[0]
    
    h_estimate = [item.clone().detach().to(device) for item in v]
    prev_h_norm = sum(h.norm().item()**2 for h in h_estimate)**0.5
    effective_batch = min(batch_size, num_train)
    effective_batch = max(effective_batch, 2)
    # ── DIAGNOSTIC: Log input magnitude ──
    v_norm = sum(vi.norm().item()**2 for vi in v)**0.5
    logger.info(f"  LiSSA input: ||v|| = {v_norm:.6e}")
    
    for i in range(recursions):
        idx = torch.randperm(num_train)[:effective_batch]
        x_batch = X_train[idx].to(device)
        y_batch = y_train[idx].to(device)
        
        model.zero_grad()
        out = model(x_batch)
        loss = F.nll_loss(out, y_batch, reduction='mean')
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        _hvp = hvps(grads, params, h_estimate)
        
        # ── DIAGNOSTIC: Log HVP magnitude at first iteration ──
        if i == 0:
            hvp_norm = sum(hv.norm().item()**2 for hv in _hvp)**0.5
            logger.info(f"  LiSSA iter 0: ||HVP|| = {hvp_norm:.6e}")
        
        with torch.no_grad():
            for j in range(len(h_estimate)):
                # Standard LiSSA update (Agarwal et al. 2017, Eq. 8):
                #   H̃⁻¹_t = v + (I − H_t/scale) × H̃⁻¹_{t−1}
                #          = v + H̃⁻¹_{t−1} − (HVP + damp·h) / scale
                h_estimate[j] = (
                    v[j].to(device)
                    + h_estimate[j]
                    - (_hvp[j] + damp * h_estimate[j]) / scale
                )
        
        del x_batch, y_batch, out, loss, grads, _hvp
        
        if (i + 1) % check_every == 0:
            curr_norm = sum(h.norm().item()**2 for h in h_estimate)**0.5
            rel_change = abs(curr_norm - prev_h_norm) / (prev_h_norm + 1e-10)
            
            if i < 100 or rel_change < convergence_tol:
                logger.info(
                    f"  LiSSA iter {i+1}: ||h|| = {curr_norm:.6e}, "
                    f"rel_change = {rel_change:.2e}"
                )
            if curr_norm > 1e6 or (i > 50 and curr_norm > prev_h_norm * 5):
                logger.warning(f" LiSSA diverging at iter {i+1}, falling back")
                h_estimate = [vi.clone().detach().to(device) / (damp * scale) for vi in v]
                break
            
            if rel_change < convergence_tol:
                break
            prev_h_norm = curr_norm
    
    # ── FIX: Do NOT divide by scale again here. ──
    # The loop already divides by scale at each step (via `/ scale` in the
    # update), so the accumulated h_estimate converges to H̃⁻¹v / scale,
    # which is the correct estimator of (H + λI)⁻¹ v (Proposition 3.5,
    # CertDNN; Proposition 3.5, Agarwal et al. 2017).
    # The previous code had an extra `h_estimate = [h / scale ...]` here,
    # producing a result that was (scale) times too small, suppressing all
    # weight updates.
    h_estimate = [h.detach() for h in h_estimate]
    
    # ── DIAGNOSTIC: Log output magnitude ──
    update_norm = sum(h.norm().item()**2 for h in h_estimate)**0.5
    logger.info(f"  LiSSA output: ||H^(-1)v|| = {update_norm:.6e}")
    
    return h_estimate


# ═══════════════════════════════════════════════════════
# NIM Fine-Tuning (separate function, called from unlearning loop)
# ═══════════════════════════════════════════════════════
def nim_finetune(model, device, X_train, y_train, agu_affected_indices,
                 lr=0.001, nim_epochs=5, pgd_c=50.0):
    """
    AGU + NIM Fine-Tuning:
    Fine-tunes the model ONLY on the nodes identified by AGU's adaptive
    neighbor selection (nodes whose features meaningfully shifted).

    FIX: Changed optimizer from SGD to Adam to match the main training loop.
    Using SGD in NIM while training uses Adam creates an inconsistency in the
    optimization landscape — gradient moments and adaptive learning rates from
    Adam's state are invalidated by SGD steps.  Uniform use of Adam throughout
    ensures consistent optimization dynamics.
    """
    if len(agu_affected_indices) < 2:
        logger.info(f"NIM/AGU: Only {len(agu_affected_indices)} affected node(s), skipping.")
        return
    
    logger.info(f"NIM/AGU: Fine-tuning on {len(agu_affected_indices)} adaptively affected nodes")

    # Fetch the dynamically affected nodes
    X_hie = X_train[agu_affected_indices].to(device)
    y_hie = y_train[agu_affected_indices].to(device)

    # Freeze BatchNorm stats to prevent corruption on small batches
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()

    # ── FIX: Use Adam (matching main training loop) instead of SGD ──
    # lr * 0.1 is a deliberately conservative NIM learning rate; Adam is used
    # because the main training loop uses Adam, and mixing optimizers creates
    # inconsistent moment estimates.
    nim_optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)

    for _ in range(nim_epochs):
        nim_optimizer.zero_grad()
        loss = F.nll_loss(model(X_hie), y_hie)
        loss.backward()
        nim_optimizer.step()

    apply_pgd_projection(model, pgd_c)
    model.eval()
    logger.info("NIM/AGU fine-tuning complete.")


def apply_lissa_update(model, inverse_hvs, device, max_update_ratio=0.01):
    """
    Apply the LiSSA weight update with norm clamping.
    
    max_update_ratio: Maximum ratio of ||update|| / ||weights||.
                      0.01 means the update can change weights by at most 1%.
    """
    with torch.no_grad():
        # Compute current weight norm
        weight_norm = sum(
            p.data.norm(2).item()**2 
            for p in model.parameters() if p.requires_grad
        )**0.5
        
        # Compute update norm
        update_norm = sum(
            h.norm().item()**2 for h in inverse_hvs
        )**0.5
        
        # Compute maximum allowed update norm
        max_update_norm = max_update_ratio * weight_norm
        
        # Clamp if necessary
        if update_norm > max_update_norm and update_norm > 0:
            clamp_ratio = max_update_norm / update_norm
            logger.info(
                f"  Clamping update: ||update||={update_norm:.4e} → "
                f"{max_update_norm:.4e} (ratio={clamp_ratio:.4e})"
            )
        else:
            clamp_ratio = 1.0
        
        # Apply update
        idx = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.data.add_(inverse_hvs[idx].to(device) * clamp_ratio)
            idx += 1
        
        logger.info(
            f"  Weight update applied: ||Δw||={update_norm*clamp_ratio:.4e}, "
            f"||w||={weight_norm:.4e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#
#   RESEARCH EVALUATION METRICS (ADDED — nothing above is modified)
#
#   1. _get_predictions()        — batch prediction helper
#   2. test_comprehensive()      — Accuracy + F1 + AUC-ROC
#   3. compute_kl_divergence()   — Fidelity: KL(Retrained || Unlearned)
#   4. compute_weight_distance() — Stability: ℓ₂ distance between models
#   5. compute_mia_score()       — Safety: entropy-based MIA
#   6. compute_theoretical_bounds() — PGD certification: grad_norm + weight_norm
#   7. compute_certification_bounds() — CertDNN full verification
#   8. retrain_gold_standard()   — Gold Standard model from scratch
#   9. compute_all_metrics()     — Master orchestrator
#  10. CSV logging helpers       — init_csv, append_csv_row, log_metrics_summary
#
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
# Helper: Extract all predictions (logits, probs, labels)
# ───────────────────────────────────────────────────────
@torch.no_grad()
def _get_predictions(model, device, X, y, batch_size):
    """
    Returns:
        all_preds:    (N,)   predicted class indices
        all_probs:    (N, C) softmax probabilities
        all_logprobs: (N, C) log-softmax outputs (raw model output)
        all_labels:   (N,)   ground truth labels
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_logprobs = []
    all_labels = []

    idx = 0
    while idx < X.shape[0]:
        x = X[idx:idx + batch_size].to(device)
        y_batch = y[idx:idx + batch_size]

        logprobs = model(x)                       # log_softmax output
        probs = torch.exp(logprobs)               # softmax probabilities
        preds = logprobs.argmax(dim=1)

        all_logprobs.append(logprobs.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

        idx += batch_size
        del x

    return (
        torch.cat(all_preds),
        torch.cat(all_probs),
        torch.cat(all_logprobs),
        torch.cat(all_labels),
    )


# ───────────────────────────────────────────────────────
# 1. Comprehensive Test: Accuracy + F1 + AUC-ROC
# ───────────────────────────────────────────────────────
@torch.no_grad()
def test_comprehensive(model, device, X, y, batch_size, evaluator=None):
    """
    Returns dict with: accuracy, f1_macro, auc_roc
    Backward-compatible: result['accuracy'] == what test() returns.
    """
    preds, probs, logprobs, labels = _get_predictions(model, device, X, y, batch_size)

    preds_np = preds.numpy()
    labels_np = labels.numpy()
    probs_np = probs.numpy()
    num_classes = probs.shape[1]

    # Accuracy
    accuracy = (preds_np == labels_np).sum() / len(labels_np)

    # Macro F1
    f1 = sklearn_f1_score(labels_np, preds_np, average='macro', zero_division="warn")

    # AUC-ROC (One-vs-Rest for multiclass)
    try:
        if num_classes == 2:
            auc_roc = sklearn_roc_auc_score(labels_np, probs_np[:, 1])
        else:
            unique_labels = np.unique(labels_np)
            if len(unique_labels) < 2:
                auc_roc = float('nan')
            else:
                auc_roc = sklearn_roc_auc_score(
                    labels_np, probs_np,
                    multi_class='ovr', average='macro'
                )
    except ValueError as e:
        logger.warning(f"AUC-ROC computation failed: {e}")
        auc_roc = float('nan')

    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'auc_roc': auc_roc,
    }


# ───────────────────────────────────────────────────────
# 2. KL Divergence (Fidelity)
# ───────────────────────────────────────────────────────
@torch.no_grad()
def compute_kl_divergence(model_unlearned, model_retrained, device,
                          X, batch_size):
    """
    Mean KL Divergence: KL(P_retrained || P_unlearned) on given data X.
    Lower KL = Higher Fidelity.
    """
    model_unlearned.eval()
    model_retrained.eval()

    total_kl = 0.0
    total_samples = 0

    idx = 0
    while idx < X.shape[0]:
        x = X[idx:idx + batch_size].to(device)

        log_p_unlearned = model_unlearned(x)
        log_p_retrained = model_retrained(x)
        p_retrained = torch.exp(log_p_retrained)

        kl = F.kl_div(
            log_p_unlearned, p_retrained,
            reduction='sum', log_target=False
        )

        total_kl += kl.item()
        total_samples += x.shape[0]
        idx += batch_size
        del x, log_p_unlearned, log_p_retrained, p_retrained

    return total_kl / max(total_samples, 1)


# ───────────────────────────────────────────────────────
# 3. Weight Distance (ℓ₂ Stability)
# ───────────────────────────────────────────────────────
@torch.no_grad()
def compute_weight_distance(model_a, model_b):
    """
    ℓ₂ Euclidean distance: ||w_a - w_b||_2
    Also returns individual weight norms.
    """
    dist_sq = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    params_a = [p for p in model_a.parameters() if p.requires_grad]
    params_b = [p for p in model_b.parameters() if p.requires_grad]

    assert len(params_a) == len(params_b), \
        f"Model parameter count mismatch: {len(params_a)} vs {len(params_b)}"

    for pa, pb in zip(params_a, params_b):
        diff = pa.data.cpu() - pb.data.cpu()
        dist_sq += (diff ** 2).sum().item()
        norm_a_sq += (pa.data ** 2).sum().item()
        norm_b_sq += (pb.data ** 2).sum().item()

    return {
        'weight_distance': math.sqrt(dist_sq),
        'weight_norm_a': math.sqrt(norm_a_sq),
        'weight_norm_b': math.sqrt(norm_b_sq),
    }


# ───────────────────────────────────────────────────────
# 4. MIA Safety Score (Entropy-based Membership Inference)
# ───────────────────────────────────────────────────────
@torch.no_grad()
def compute_mia_score(model, device, X_forgotten, y_forgotten,
                      X_test, y_test, batch_size, max_samples=500):
    """
    Entropy-based MIA. Attacker tries to distinguish "forgotten" nodes
    from "test" nodes using prediction entropy.
    
    Returns mia_auc:
        0.5 = perfect safety (random guessing)
        1.0 = complete failure (attacker wins)
    """
    model.eval()

    def _compute_entropy(X, n_samples):
        n = min(n_samples, X.shape[0])
        if n == 0:
            return np.array([])

        perm = torch.randperm(X.shape[0])[:n]
        X_sub = X[perm]

        entropies = []
        idx_inner = 0
        while idx_inner < n:
            x = X_sub[idx_inner:idx_inner + batch_size].to(device)
            logprobs = model(x)
            probs = torch.exp(logprobs)
            ent = -(probs * logprobs).sum(dim=1)
            entropies.append(ent.cpu().numpy())
            idx_inner += batch_size
            del x, logprobs, probs

        return np.concatenate(entropies)

    ent_forgotten = _compute_entropy(X_forgotten, max_samples)
    ent_test = _compute_entropy(X_test, max_samples)

    if len(ent_forgotten) == 0 or len(ent_test) == 0:
        logger.warning("MIA: Insufficient samples, returning NaN")
        return float('nan')

    scores = np.concatenate([ent_forgotten, ent_test])
    labels = np.concatenate([
        np.ones(len(ent_forgotten)),
        np.zeros(len(ent_test)),
    ])
    attack_scores = -scores  # lower entropy → more confident → likely member

    try:
        mia_auc = sklearn_roc_auc_score(labels, attack_scores)
    except ValueError as e:
        logger.warning(f"MIA AUC computation failed: {e}")
        mia_auc = float('nan')

    return mia_auc


# ───────────────────────────────────────────────────────
# 5. Theoretical Bounds (Gradient Norm + Weight Norm)
# ───────────────────────────────────────────────────────
def compute_theoretical_bounds(model, device, X_train, y_train, batch_size):
    """
    Computes grad_norm and weight_norm for PGD certification.
    """
    grads = cal_grad_handloader(
        model, device, X_train, y_train, batch_size, retain=False
    )
    grad_norm = sum(g.norm().item()**2 for g in grads) ** 0.5
    del grads

    weight_norm = sum(
        p.data.norm(2).item()**2
        for p in model.parameters() if p.requires_grad
    ) ** 0.5

    return {
        'grad_norm': grad_norm,
        'weight_norm': weight_norm,
    }


# ───────────────────────────────────────────────────────
# 6. CertDNN: Certified Unlearning Verification
# ───────────────────────────────────────────────────────
def compute_certification_bounds(
    model_unlearned, model_retrained, device,
    X_train, y_train, batch_size,
    edges_removed,
    deg_fn,
    train_size,
    feat_dim,
    lam,
    rmax,
    num_nodes,
    prop_step,
    pgd_c=50.0,
    removal_type='edge',
):
    """
    CertDNN: computes theoretical certification radius and checks
    whether actual weight distance stays within it.
    
    Returns dict with theoretical bounds, actual distance, PGD compliance,
    and certification verdict (PASS/FAIL).
    """
    from utils import (
        get_worst_Gbound_edge,
        get_worst_Gbound_node,
        get_worst_Gbound_feat,
    )

    cert = {}

    # ── 1. Theoretical Bounds ──
    if removal_type == 'edge':
        approx_norms = []
        unlearn_norms = []
        for edge in edges_removed:
            u, v = edge[0], edge[1]
            deg_u = max(deg_fn(u), 1)
            deg_v = max(deg_fn(v), 1)
            a_norm, u_norm = get_worst_Gbound_edge(
                deg_u, deg_v, train_size, feat_dim, lam, rmax, num_nodes, prop_step
            )
            approx_norms.append(a_norm)
            unlearn_norms.append(u_norm)

        cert['theoretical_approx_bound'] = sum(approx_norms)
        cert['theoretical_unlearn_bound'] = sum(unlearn_norms)
        cert['theoretical_total_bound'] = cert['theoretical_approx_bound'] + cert['theoretical_unlearn_bound']

    elif removal_type == 'node':
        degs = [max(deg_fn(n), 1) for n in edges_removed]
        a_norm, u_norm = get_worst_Gbound_node(
            degs, train_size, feat_dim, lam, rmax, num_nodes, prop_step
        )
        cert['theoretical_approx_bound'] = a_norm
        cert['theoretical_unlearn_bound'] = u_norm
        cert['theoretical_total_bound'] = a_norm + u_norm

    elif removal_type == 'feature':
        _deg = max(deg_fn(edges_removed[0]), 1) if len(edges_removed) > 0 else 1
        a_norm, u_norm = get_worst_Gbound_feat(
            _deg, train_size, feat_dim, lam, rmax, num_nodes, prop_step
        )
        cert['theoretical_approx_bound'] = a_norm
        cert['theoretical_unlearn_bound'] = u_norm
        cert['theoretical_total_bound'] = a_norm + u_norm

    # ── 2. Actual Weight Distance ──
    if model_retrained is not None:
        wd = compute_weight_distance(model_unlearned, model_retrained)
        cert['actual_weight_distance'] = wd['weight_distance']
    else:
        cert['actual_weight_distance'] = float('nan')

    # ── 3. Gradient Residual ──
    grads = cal_grad_handloader(
        model_unlearned, device, X_train, y_train, batch_size, retain=False
    )
    cert['grad_residual_norm'] = sum(g.norm().item()**2 for g in grads) ** 0.5
    del grads

    # ── 4. Weight Norm ──
    cert['cert_weight_norm'] = sum(
        p.data.norm(2).item()**2
        for p in model_unlearned.parameters() if p.requires_grad
    ) ** 0.5

    # ── 5. Per-Parameter PGD Check ──
    pgd_violations = 0
    max_param_norm = 0.0
    for p in model_unlearned.parameters():
        if p.requires_grad:
            p_norm = p.data.norm(2).item()
            max_param_norm = max(max_param_norm, p_norm)
            if p_norm > pgd_c * (1 + 1e-6):
                pgd_violations += 1

    cert['pgd_max_param_norm'] = max_param_norm
    cert['pgd_violations'] = pgd_violations
    cert['pgd_compliant'] = (pgd_violations == 0)

    # ── 6. Certification Verdict ──
    if not math.isnan(cert['actual_weight_distance']):
        cert['certified'] = (
            cert['actual_weight_distance'] <= cert['theoretical_total_bound']
            and cert['pgd_compliant']
        )
        cert['certification_ratio'] = (
            cert['actual_weight_distance'] / max(cert['theoretical_total_bound'], 1e-15)
        )
    else:
        cert['certified'] = None
        cert['certification_ratio'] = float('nan')

    return cert


def log_certification_summary(cert, batch_idx, logger_ref=None):
    """Pretty-print CertDNN results."""
    _log = logger_ref or logger

    status = "CERTIFIED" if cert.get('certified') else "NOT CERTIFIED"
    if cert.get('certified') is None:
        status = "UNKNOWN (no retrained model)"

    _log.info(f"{'─'*60}")
    _log.info(f"  CERTDNN VERIFICATION — Batch {batch_idx} — {status}")
    _log.info(f"{'─'*60}")
    _log.info(f"  [Theoretical Bounds]")
    _log.info(f"    Approximation Bound:  {cert.get('theoretical_approx_bound', float('nan')):.6e}")
    _log.info(f"    Unlearning Bound:     {cert.get('theoretical_unlearn_bound', float('nan')):.6e}")
    _log.info(f"    Total Bound:          {cert.get('theoretical_total_bound', float('nan')):.6e}")
    _log.info(f"  [Actual Measurements]")
    _log.info(f"    Weight Distance:      {cert.get('actual_weight_distance', float('nan')):.6e}")
    _log.info(f"    Gradient Residual:    {cert.get('grad_residual_norm', float('nan')):.6e}")
    _log.info(f"    Weight Norm:          {cert.get('cert_weight_norm', float('nan')):.4f}")
    _log.info(f"  [PGD Compliance]")
    _log.info(f"    Max Param Norm:       {cert.get('pgd_max_param_norm', float('nan')):.4f}")
    _log.info(f"    Violations:           {cert.get('pgd_violations', 'N/A')}")
    _log.info(f"    Compliant:            {cert.get('pgd_compliant', 'N/A')}")
    _log.info(f"  [Verdict]")
    _log.info(f"    Ratio (actual/bound): {cert.get('certification_ratio', float('nan')):.4f}")
    _log.info(f"    RESULT:               {status}")
    _log.info(f"{'─'*60}")


# ───────────────────────────────────────────────────────
# 7. Gold Standard Retraining
# ───────────────────────────────────────────────────────
def retrain_gold_standard(model_class, model_kwargs, device,
                          X_train, y_train, X_val, y_val,
                          batch_size, lr, wd, epochs, patience,
                          checkpt_file, evaluator=None,
                          noises=None, pgd_c=50.0):
    """
    Retrain a fresh model from scratch on the REMAINING data.
    Returns (retrained_model, retrain_time).
    """
    logger.info("--- Retraining Gold Standard Model ---")

    retrained_model = model_class(**model_kwargs).to(device)
    retrained_model.reset_parameters()

    retrain_optimizer = optim.Adam(
        retrained_model.parameters(), lr=lr, weight_decay=wd
    )

    retrain_time = train_model(
        retrained_model, device, X_train, y_train, batch_size,
        retrain_optimizer, epochs, X_val, y_val, evaluator,
        checkpt_file, patience, verbose=False,
        noises=noises, pgd_c=pgd_c
    )

    retrained_model.load_state_dict(
        torch.load(checkpt_file, map_location=device)
    )

    logger.info(f"Gold Standard retrain time: {retrain_time:.2f}s")
    return retrained_model, retrain_time


# ───────────────────────────────────────────────────────
# 8. MASTER: Compute All Metrics
# ───────────────────────────────────────────────────────
def compute_all_metrics(
    model_unlearned, device,
    X_test, y_test,
    X_train_remaining, y_train_remaining,
    batch_size,
    model_retrained=None,
    X_forgotten=None, y_forgotten=None,
    mia_max_samples=500,
    compute_bounds=True,
):
    """
    Master function: computes ALL research metrics in one call.
    Returns dict with all metrics.
    """
    metrics = {}

    # ── 1. Utility (Accuracy, F1, AUC-ROC) on test set ──
    logger.info("  Computing Utility metrics (Acc, F1, AUC-ROC)...")
    utility = test_comprehensive(
        model_unlearned, device, X_test, y_test, batch_size
    )
    metrics['test_accuracy'] = utility['accuracy']
    metrics['test_f1_macro'] = utility['f1_macro']
    metrics['test_auc_roc'] = utility['auc_roc']

    # ── 2. Fidelity (KL Divergence) ──
    if model_retrained is not None:
        logger.info("  Computing KL Divergence (Fidelity)...")
        kl = compute_kl_divergence(
            model_unlearned, model_retrained, device,
            X_train_remaining, batch_size
        )
        metrics['kl_divergence'] = kl

        retrained_utility = test_comprehensive(
            model_retrained, device, X_test, y_test, batch_size
        )
        metrics['retrained_accuracy'] = retrained_utility['accuracy']
        metrics['retrained_f1_macro'] = retrained_utility['f1_macro']
        metrics['retrained_auc_roc'] = retrained_utility['auc_roc']
    else:
        metrics['kl_divergence'] = float('nan')
        metrics['retrained_accuracy'] = float('nan')
        metrics['retrained_f1_macro'] = float('nan')
        metrics['retrained_auc_roc'] = float('nan')

    # ── 3. Stability (Weight Distance) ──
    if model_retrained is not None:
        logger.info("  Computing Weight Distance (Stability)...")
        wd = compute_weight_distance(model_unlearned, model_retrained)
        metrics['weight_distance'] = wd['weight_distance']
        metrics['unlearned_weight_norm'] = wd['weight_norm_a']
        metrics['retrained_weight_norm'] = wd['weight_norm_b']
    else:
        metrics['weight_distance'] = float('nan')
        metrics['unlearned_weight_norm'] = float('nan')
        metrics['retrained_weight_norm'] = float('nan')

    # ── 4. Safety (MIA) ──
    if X_forgotten is not None and y_forgotten is not None and X_forgotten.shape[0] > 0:
        logger.info("  Computing MIA Safety Score...")
        mia = compute_mia_score(
            model_unlearned, device,
            X_forgotten, y_forgotten,
            X_test, y_test,
            batch_size, max_samples=mia_max_samples
        )
        metrics['mia_safety_auc'] = mia
    else:
        metrics['mia_safety_auc'] = float('nan')

    # ── 5. Theoretical Bounds ──
    if compute_bounds:
        logger.info("  Computing Theoretical Bounds (grad_norm, weight_norm)...")
        bounds = compute_theoretical_bounds(
            model_unlearned, device,
            X_train_remaining, y_train_remaining, batch_size
        )
        metrics['grad_norm'] = bounds['grad_norm']
        metrics['weight_norm'] = bounds['weight_norm']
    else:
        metrics['grad_norm'] = float('nan')
        metrics['weight_norm'] = float('nan')

    return metrics


# ───────────────────────────────────────────────────────
# 9. CSV Summary Sheet
# ───────────────────────────────────────────────────────

CSV_COLUMNS = [
    'batch_idx',
    # Timing
    'unlearn_time', 'update_time', 'total_time', 'retrain_time',
    # Utility — Unlearned
    'test_accuracy', 'test_f1_macro', 'test_auc_roc',
    # Utility — Retrained (Gold Standard)
    'retrained_accuracy', 'retrained_f1_macro', 'retrained_auc_roc',
    # Fidelity
    'kl_divergence',
    # Stability
    'weight_distance', 'unlearned_weight_norm', 'retrained_weight_norm',
    # Safety
    'mia_safety_auc',
    # Theoretical Bounds
    'grad_norm', 'weight_norm',
    # CertDNN
    'theoretical_approx_bound', 'theoretical_unlearn_bound',
    'theoretical_total_bound',
    'actual_weight_distance', 'grad_residual_norm', 'cert_weight_norm',
    'pgd_max_param_norm', 'pgd_violations', 'pgd_compliant',
    'certification_ratio', 'certified',
]


def init_csv(csv_path):
    """Initialize the CSV file with headers."""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
    logger.info(f"CSV Summary Sheet initialized: {csv_path}")


def append_csv_row(csv_path, row_dict):
    """Append one row to the CSV. Missing keys filled with NaN."""
    clean_row = {}
    for col in CSV_COLUMNS:
        val = row_dict.get(col, float('nan'))
        if isinstance(val, float):
            clean_row[col] = f"{val:.6f}" if not math.isnan(val) else "NaN"
        elif isinstance(val, bool):
            clean_row[col] = str(val)
        elif isinstance(val, int):
            clean_row[col] = str(val)
        else:
            clean_row[col] = str(val)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(clean_row)


def log_metrics_summary(metrics, batch_idx, logger_ref=None):
    """Pretty-print a metrics dictionary."""
    _log = logger_ref or logger
    _log.info(f"{'='*60}")
    _log.info(f"  METRICS SUMMARY — Batch {batch_idx}")
    _log.info(f"{'='*60}")
    _log.info(f"  [Utility]")
    _log.info(f"    Accuracy:    {metrics.get('test_accuracy', float('nan')):.4f}")
    _log.info(f"    F1 (Macro):  {metrics.get('test_f1_macro', float('nan')):.4f}")
    _log.info(f"    AUC-ROC:     {metrics.get('test_auc_roc', float('nan')):.4f}")
    _log.info(f"  [Fidelity]")
    _log.info(f"    KL Divergence:        {metrics.get('kl_divergence', float('nan')):.6f}")
    _log.info(f"  [Stability]")
    _log.info(f"    Weight Distance:      {metrics.get('weight_distance', float('nan')):.6f}")
    _log.info(f"    Unlearned ||w||:      {metrics.get('unlearned_weight_norm', float('nan')):.4f}")
    _log.info(f"    Retrained ||w||:      {metrics.get('retrained_weight_norm', float('nan')):.4f}")
    _log.info(f"  [Safety]")
    _log.info(f"    MIA AUC:              {metrics.get('mia_safety_auc', float('nan')):.4f}")
    _log.info(f"  [Theoretical Bounds]")
    _log.info(f"    ||grad||:             {metrics.get('grad_norm', float('nan')):.6e}")
    _log.info(f"    ||w||:                {metrics.get('weight_norm', float('nan')):.4f}")

    if not math.isnan(metrics.get('retrained_accuracy', float('nan'))):
        _log.info(f"  [Gold Standard]")
        _log.info(f"    Retrained Acc:  {metrics['retrained_accuracy']:.4f}")
        _log.info(f"    Retrained F1:   {metrics['retrained_f1_macro']:.4f}")
        _log.info(f"    Retrained AUC:  {metrics['retrained_auc_roc']:.4f}")
    _log.info(f"{'='*60}")