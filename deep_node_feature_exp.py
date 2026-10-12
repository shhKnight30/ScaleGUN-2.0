import numpy as np
import argparse
import propagation
import uuid
import torch
from utils import *
from deep_unlearn_utils import *
import torch.nn.functional as F
import time
import gc
import math
from datetime import datetime
import logging
import pytz
from model import ClassMLP
from ogb.nodeproppred import Evaluator
from argparser import argparser
import random
import os

torch.set_printoptions(precision=10)
name = "deep_node_feat"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format="%(filename)s - %(levelname)s - %(message)s", handlers=[])
setup_logger(name)
setup_unlearn_logger(name)

def clear_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def main():
    args = argparser()
    seed_everything(seeds[args.seed])
    tz = pytz.timezone("Asia/Kolkata")
    dt = datetime.now(tz).strftime("%m%d_%H%M")
    set_logger(args, logger, dt, name)
    logger.info(args)

    # ═══════════════════════════════════════════════════
    # Safe access helpers for NEW args (backward-compatible)
    # ═══════════════════════════════════════════════════
    do_metrics      = getattr(args, 'compute_metrics', False)
    do_retrain      = getattr(args, 'retrain', False)
    do_certdnn      = getattr(args, 'use_certdnn', False)
    do_agu          = getattr(args, 'agu_check', False)
    do_nim          = getattr(args, 'use_nim', False)
    csv_output      = getattr(args, 'csv_output', '')
    metric_every    = getattr(args, 'metric_every', 1)
    mia_num_samples = getattr(args, 'mia_num_samples', 500)
    pgd_c           = getattr(args, 'pgd_c', 5.0)
    pgd_mode        = getattr(args, 'pgd_mode', 'per_param')
    nim_epochs      = getattr(args, 'nim_epochs', 5)
    retrain_epochs  = getattr(args, 'retrain_epochs', args.epochs)
    retrain_patience= getattr(args, 'retrain_patience', args.patience)
    save_retrained  = getattr(args, 'save_retrained_model', False)
    cert_removal    = getattr(args, 'cert_removal_type', 'node') # default to node
    lissa_bs        = getattr(args, 'lissa_batch_size', 256)
    lissa_rec       = getattr(args, 'lissa_recursions', 200)
    lissa_damp      = getattr(args, 'lissa_damp', 0.5)
    lissa_scale     = getattr(args, 'lissa_scale', 50.0)
    lissa_tol       = getattr(args, 'lissa_convergence_tol', 1e-4)
    lissa_check     = getattr(args, 'lissa_check_every', 50)
    max_update_r    = getattr(args, 'max_update_ratio', 0.01)

    prefix_args = f"Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_layer_{args.layer}_removal_{args.removal_mode}"

    # Auto-generate CSV path if empty
    if do_metrics and csv_output == '':
        csv_output = (
            f"{args.analysis_path}/{args.dataset}/"
            f"metrics_summary_{prefix_args}_seed{args.seed}.csv"
        )

    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result")
    base_path = f"{args.analysis_path}/{args.dataset}/{name}_result/{prefix_args}"

    f_tot_cost = open(base_path + "_cost.txt", "ab")
    f_unlearn_cost = open(base_path + "_unlearn_cost.txt", "ab")
    f_update_cost = open(base_path + "_update_cost.txt", "ab")
    f_acc = open(base_path + "_acc.txt", "ab")

    if args.dev > -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.dev}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ═══════════════════════════════════════════════════
    # Initialize CSV Summary Sheet
    # ═══════════════════════════════════════════════════
    if do_metrics:
        check_dir(os.path.dirname(csv_output) if os.path.dirname(csv_output) else '.')
        init_csv(csv_output)
        logger.info(f"Research metrics ENABLED → CSV: {csv_output}")
        if do_retrain:
            logger.info("Gold Standard retraining ENABLED")
        if do_certdnn:
            logger.info("CertDNN verification ENABLED")
    else:
        logger.info("Research metrics DISABLED (use --compute_metrics to enable)")

    start = time.perf_counter()
    data, edge_index = load_data(args.path, args.dataset)
    weights = get_prop_weight(args.weight_mode, args.prop_step, args.decay)

    data.y = data.y.long()
    full_y = data.y.clone()
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_deep_split_large(
        data, args.train_mode, args.Y_binary
    )

    feat = preprocess_data(data.x, axis_num=args.axis_num)
    del data
    column_sum_avg = feat.abs().sum(axis=0).mean()

    init_rmax = args.rmax * column_sum_avg
    update_rmax = init_rmax
    args.rmax = init_rmax

    feat = feat.T
    origin_embedding = np.copy(feat.numpy())

    # FIX 1: Force C-contiguous
    origin_embedding = np.ascontiguousarray(origin_embedding, dtype=np.float64)

    # FIX 2: Use transpose version
    g = propagation.InstantGNN_transpose()

    del_path = os.path.join(args.path, args.del_path_suffix)
    prop_time = g.init_push_graph(
        del_path, args.dataset, origin_embedding, edge_index.T,
        args.prop_step, args.r, weights, args.num_threads,
        init_rmax, origin_embedding.shape[0]
    )
    del edge_index
    gc.collect()

    X = torch.FloatTensor(origin_embedding.T)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    del X

    model_kwargs = {
        'in_channels': feat_dim,
        'hidden_channels': args.hidden,
        'out_channels': num_classes,
        'num_layers': args.layer,
        'dropout': args.dropout,
    }

    model = ClassMLP(feat_dim, args.hidden, num_classes, args.layer, args.dropout).to(device)

    b_std = 0 if getattr(args, 'compare_gnorm', False) else args.std
    noises = [
        b_std * torch.randn(p.shape).float().to(device)
        for p in model.parameters() if p.requires_grad
    ]

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    evaluator = Evaluator(name=args.dataset) if args.dataset in ["ogbn-arxiv", "ogbn-products"] else None

    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model")
    checkpt_file = f"{args.analysis_path}/{args.dataset}/{name}_model/{prefix_args}_init.pt"
    model.reset_parameters()

    logger.info("--- Starting Base Model Training ---")
    train_time = train_model(
        model, device, X_train, y_train, args.train_batch, optimizer,
        args.epochs, X_val, y_val, evaluator, checkpt_file, args.patience,
        verbose=True, noises=noises, pgd_c=pgd_c
    )

    model.load_state_dict(torch.load(checkpt_file, map_location=device))
    test_acc = test(model, device, X_test, y_test, args.test_batch, evaluator)
    val_acc = test(model, device, X_val, y_val, args.test_batch, evaluator)

    logger.info(f"Base Train cost: {train_time:.2f}s | Test Accuracy: {100*test_acc:.2f}%")

    original_checkpt = f"{args.analysis_path}/{args.dataset}/{name}_model/{prefix_args}_original.pt"
    torch.save(model.state_dict(), original_checkpt)

    if do_metrics:
        logger.info("--- Computing Base Model Metrics ---")
        base_metrics = compute_all_metrics(
            model_unlearned=model, device=device,
            X_test=X_test, y_test=y_test,
            X_train_remaining=X_train, y_train_remaining=y_train,
            batch_size=args.test_batch, model_retrained=None,
            X_forgotten=None, y_forgotten=None,
            mia_max_samples=mia_num_samples, compute_bounds=True,
        )
        base_metrics['batch_idx'] = -1
        base_metrics['unlearn_time'] = 0.0
        base_metrics['update_time'] = prop_time
        base_metrics['total_time'] = train_time + prop_time
        base_metrics['retrain_time'] = train_time
        log_metrics_summary(base_metrics, batch_idx=-1, logger_ref=logger)
        append_csv_row(csv_output, base_metrics)

    acc_removal = [[val_acc], [test_acc]]
    old_grad = cal_grad_handloader(model, device, X_train, y_train, args.test_batch, retain=False)

    update_cost, unlearn_cost, tot_cost = [prop_time], [train_time], [train_time + prop_time]

    node_file = os.path.join(del_path, args.dataset, f"{args.dataset}_del_nodes.npy")
    del_nodes = np.load(node_file).flatten()
    np.random.seed(args.seed)
    np.random.shuffle(del_nodes)

    all_forgotten_nodes = set()
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    train_idx_to_pos = {idx.item(): pos for pos, idx in enumerate(train_indices)}

    del X_train, X_val, X_test, noises
    gc.collect()

    logger.info("*" * 20)
    logger.info(f"--- Starting Unlearning ({args.removal_mode} Removal) Loop ---")
    start_time = time.perf_counter()
    node_idx_start = args.edge_idx_start

    # Keep a reference to original full_X strictly for MIA extraction
    origin_X_full_static = torch.FloatTensor(origin_embedding.T)

    for i in range(args.num_batch_removes):
        nodes = del_nodes[
            node_idx_start + i * args.num_removes :
            node_idx_start + args.num_removes * (i + 1)
        ].tolist()

        for node in nodes:
            all_forgotten_nodes.add(node)

        # ═══════════════════════════════════════════════════
        # AGU FIX 1: Cache the FULL features before update
        # ═══════════════════════════════════════════════════
        X_old_full = torch.FloatTensor(origin_embedding.T)

        if args.removal_mode == "node":
            return_time = g.UpdateNodes(nodes, origin_embedding, args.num_threads, update_rmax)
        elif args.removal_mode == "feature":
            return_time = g.UpdateFeatures(nodes, origin_embedding, args.num_threads, update_rmax)

        update_cost.append(return_time)
        update_finish_time = time.perf_counter()

        # ═══════════════════════════════════════════════════
        # MIA Prep: Extract features BEFORE shrinking the mask
        # ═══════════════════════════════════════════════════
        X_forgotten_batch = None
        y_forgotten_batch = None
        if do_metrics:
            valid_forgotten_nodes = [n for n in nodes if train_mask[n]]
            if len(valid_forgotten_nodes) > 0:
                X_forgotten_batch = origin_X_full_static[valid_forgotten_nodes].to(device)
                y_forgotten_batch = full_y[valid_forgotten_nodes].to(device)

        # CRITICAL: Drop nodes from training mask AFTER caching old features
        train_mask[nodes] = False
        y_train_updated = full_y[train_mask].to(device)

        # ═══════════════════════════════════════════════════
        # AGU FIX 2 & 3: Extract new features & AGU Selection
        # ═══════════════════════════════════════════════════
        X_new = torch.FloatTensor(origin_embedding.T)

        # We must slice BOTH old and new features using the NEW train_mask
        # to ensure dimensions match exactly for comparison
        X_old_train_eval = X_old_full[train_mask]
        X_new_train = X_new[train_mask]

        feature_diffs = torch.norm(X_new_train - X_old_train_eval, dim=1)
        agu_affected_indices = (feature_diffs > 1e-6).nonzero(as_tuple=True)[0]
        del X_old_full, X_old_train_eval

        X_new_val = X_new[val_mask]
        X_new_test = X_new[test_mask]
        del X_new

        model_params = [p for p in model.parameters() if p.requires_grad]
        new_grad = cal_grad_handloader(
            model, device, X_new_train, y_train_updated, args.test_batch, retain=True
        )

        vs = tuple((old_grad[k] - new_grad[k]).detach() for k in range(len(old_grad)))
        del old_grad, new_grad
        gc.collect()

        v_norm = sum(vi.norm().item()**2 for vi in vs)**0.5
        if v_norm < 1e-10:
            logger.info(f" ||v|| =0 , skipping LiSSA for batch {i}")
            inverse_hvs = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
        else:
            logger.info(f"Running LiSSA Approximation for removal batch {i}....")
            inverse_hvs = lissa_inverse_hvp(
                v=vs, model=model, device=device,
                X_train=X_new_train, y_train=y_train_updated,
                batch_size=lissa_bs, recursions=lissa_rec,
                damp=lissa_damp, scale=lissa_scale,
                convergence_tol=lissa_tol, check_every=lissa_check
            )

        apply_lissa_update(model, inverse_hvs, device, max_update_ratio=max_update_r)
        apply_pgd_projection(model, pgd_c=pgd_c, mode=pgd_mode)

        if args.std > 0:
            noise_scale = args.std / math.sqrt(i + 2)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        per_param_grad = noise_scale / math.sqrt(p.numel())
                        p.data.add_(torch.randn_like(p) * per_param_grad)

        # ═══════════════════════════════════════════════════
        # AGU FIX 4: NIM Fine-tuning on Affected Nodes
        # ═══════════════════════════════════════════════════
        if do_nim:
            nim_finetune(
                model, device, X_new_train, y_train_updated,
                agu_affected_indices, lr=args.lr,
                nim_epochs=nim_epochs, pgd_c=pgd_c
            )

        del model_params, inverse_hvs, vs
        gc.collect()
        clear_cache(device)

        old_grad = cal_grad_handloader(
            model, device, X_new_train, y_train_updated, args.test_batch, retain=False
        )

        # Standalone AGU Diagnostics
        if do_agu and not do_metrics:
            _agu_wnorm = sum(p.data.norm(2).item()**2 for p in model.parameters() if p.requires_grad) ** 0.5
            _agu_gnorm = sum(g_tensor.norm().item()**2 for g_tensor in old_grad) ** 0.5
            _agu_pgd_ok = all(p.data.norm(2).item() <= pgd_c * (1 + 1e-6) for p in model.parameters() if p.requires_grad)
            logger.info(f"  [AGU] Batch {i}: ||grad||={_agu_gnorm:.6e}, ||w||={_agu_wnorm:.4f}, PGD_ok={_agu_pgd_ok}")

        test_acc = test(model, device, X_new_test, y_test, args.test_batch, evaluator)
        val_acc = test(model, device, X_new_val, y_val, args.test_batch, evaluator)

        remove_finish_time = time.perf_counter()
        acc_removal[0].append(val_acc)
        acc_removal[1].append(test_acc)

        unlearn_time_i = remove_finish_time - update_finish_time
        total_time_i = unlearn_time_i + return_time
        unlearn_cost.append(unlearn_time_i)
        tot_cost.append(total_time_i)

        checkpt_file = f"{args.analysis_path}/{args.dataset}/{name}_model/{prefix_args}_{i}.pt"
        torch.save(model.state_dict(), checkpt_file)

        # ═══════════════════════════════════════════════════
        # Metrics & Retraining Integration
        # ═══════════════════════════════════════════════════
        if do_metrics and (i % metric_every == 0 or i == args.num_batch_removes - 1):
            logger.info(f"--- Computing Research Metrics for Batch {i} ---")

            model_retrained = None
            retrain_time_i = float('nan')

            if do_retrain:
                retrain_checkpt = f"{args.analysis_path}/{args.dataset}/{name}_model/retrained_batch_{i}.pt"
                retrain_noises = None
                if b_std > 0:
                    tmp_model = ClassMLP(**model_kwargs).to(device)
                    retrain_noises = [
                        b_std * torch.randn(p.shape).float().to(device)
                        for p in tmp_model.parameters() if p.requires_grad
                    ]
                    del tmp_model

                model_retrained, retrain_time_i = retrain_gold_standard(
                    model_class=ClassMLP, model_kwargs=model_kwargs, device=device,
                    X_train=X_new_train, y_train=y_train_updated, X_val=X_new_val, y_val=y_val,
                    batch_size=args.train_batch, lr=args.lr, wd=args.wd, epochs=retrain_epochs,
                    patience=retrain_patience, checkpt_file=retrain_checkpt, evaluator=evaluator,
                    noises=retrain_noises, pgd_c=pgd_c,
                )
                if save_retrained:
                    logger.info(f"Retrained model saved: {retrain_checkpt}")
                del retrain_noises
                gc.collect()
                clear_cache(device)

            metrics = compute_all_metrics(
                model_unlearned=model, device=device, X_test=X_new_test, y_test=y_test,
                X_train_remaining=X_new_train, y_train_remaining=y_train_updated,
                batch_size=args.test_batch, model_retrained=model_retrained,
                X_forgotten=X_forgotten_batch, y_forgotten=y_forgotten_batch,
                mia_max_samples=mia_num_samples, compute_bounds=True,
            )

            if do_certdnn:
                def deg_fn(node_id):
                    try: return max(int(g.GetDegree(int(node_id))), 1)
                    except Exception: return 1

                cert_metrics = compute_certification_bounds(
                    model_unlearned=model, model_retrained=model_retrained, device=device,
                    X_train=X_new_train, y_train=y_train_updated, batch_size=args.test_batch,
                    edges_removed=nodes, deg_fn=deg_fn, train_size=X_new_train.shape[0],
                    feat_dim=feat_dim, lam=args.lam, rmax=args.rmax, num_nodes=origin_embedding.shape[1],
                    prop_step=args.prop_step, pgd_c=pgd_c, removal_type=args.removal_mode,
                )
                metrics.update(cert_metrics)

            metrics['batch_idx'] = i
            metrics['unlearn_time'] = unlearn_time_i
            metrics['update_time'] = return_time
            metrics['total_time'] = total_time_i
            metrics['retrain_time'] = retrain_time_i

            log_metrics_summary(metrics, batch_idx=i, logger_ref=logger)
            append_csv_row(csv_output, metrics)

            if model_retrained is not None:
                if not save_retrained:
                    try: os.remove(retrain_checkpt)
                    except OSError: pass
                del model_retrained
                gc.collect()
                clear_cache(device)

        del X_new_train, X_new_val, X_new_test
        if 'X_forgotten_batch' in locals():
            del X_forgotten_batch, y_forgotten_batch

        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: {args.removal_mode} del = {nodes[0]}, "
                f"Val acc = {acc_removal[0][-1]:.4f}, Test acc = {acc_removal[1][-1]:.4f}, "
                f"avg update cost: {np.mean(update_cost[1:]):.4f}, avg unlearn cost: {np.mean(unlearn_cost[1:]):.4f}"
            )

    logger.info(f"Total Unlearning Process Cost: {time.perf_counter() - start_time:.4f}s")

    np.savetxt(f_tot_cost, tot_cost, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")
    f_tot_cost.close(); f_unlearn_cost.close(); f_update_cost.close(); f_acc.close()

    if do_metrics:
        logger.info(f"{'='*60}")
        logger.info(f"  CSV Summary Sheet: {csv_output}")
        logger.info(f"  Total batches:     {args.num_batch_removes}")
        logger.info(f"  Forgotten nodes:   {len(all_forgotten_nodes)}")
        logger.info(f"  Final test acc:    {acc_removal[1][-1]:.4f}")
        logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()