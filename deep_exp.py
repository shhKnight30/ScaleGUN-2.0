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
name = "deep_edge"
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
    cert_removal    = getattr(args, 'cert_removal_type', 'edge')
    lissa_bs        = getattr(args, 'lissa_batch_size', 256)
    lissa_rec       = getattr(args, 'lissa_recursions', 200)
    lissa_damp      = getattr(args, 'lissa_damp', 0.5)
    lissa_scale     = getattr(args, 'lissa_scale', 50.0)
    lissa_tol       = getattr(args, 'lissa_convergence_tol', 1e-4)
    lissa_check     = getattr(args, 'lissa_check_every', 50)
    max_update_r    = getattr(args, 'max_update_ratio', 0.01)

    # Auto-generate CSV path if empty
    if do_metrics and csv_output == '':
        csv_output = (
            f"{args.analysis_path}/{args.dataset}/"
            f"metrics_summary_Batch{args.num_batch_removes}_"
            f"Num{args.num_removes}_seed{args.seed}.csv"
        )

    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result")
    base_path = (
        f"{args.analysis_path}/{args.dataset}/{name}_result/"
        f"Batch_{args.num_batch_removes}_Num_{args.num_removes}_"
        f"lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_"
        f"rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_"
        f"layer_{args.layer}_batch_{args.train_batch}_"
        f"drop_{args.dropout}_hidden_{args.hidden}"
    )

    f_tot_cost = open(base_path + "_cost.txt", "ab")
    f_unlearn_cost = open(base_path + "_unlearn_cost.txt", "ab")
    f_update_cost = open(base_path + "_update_cost.txt", "ab")
    f_acc = open(base_path + "_acc.txt", "ab")
    origin_rmax = args.rmax

    if args.dev > -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.dev}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ═══════════════════════════════════════════════════
    # NEW: Initialize CSV Summary Sheet (if metrics enabled)
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
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_deep_split_large(
        data, args.train_mode, args.Y_binary
    )

    feat = preprocess_data(data.x, axis_num=args.axis_num)
    del data
    column_sum_avg = feat.abs().sum(axis=0).mean()
    logger.info(f"column_sum_avg: {column_sum_avg}")

    init_rmax = args.rmax * column_sum_avg
    update_rmax = init_rmax
    logger.info(f"Init rmax: {init_rmax:.6e}, Update rmax: {update_rmax:.6e}")
    args.rmax = init_rmax

    feat = feat.T
    origin_embedding = np.copy(feat.numpy())

    # ═══════════════════════════════════════════════════
    # FIX 1: Force C-contiguous for Eigen compatibility
    # ═══════════════════════════════════════════════════
    origin_embedding = np.ascontiguousarray(origin_embedding, dtype=np.float64)
    logger.info(
        f"origin_embedding: shape={origin_embedding.shape}, "
        f"c_contiguous={origin_embedding.flags.c_contiguous}, "
        f"dtype={origin_embedding.dtype}"
    )

    # ═══════════════════════════════════════════════════
    # FIX 2: Use transpose version — has explicit copy-back logic
    # ═══════════════════════════════════════════════════
    g = propagation.InstantGNN_transpose()
    logger.info(f"Using InstantGNN_transpose")

    del_path = os.path.join(args.path, args.del_path_suffix)
    prop_time = g.init_push_graph(
        del_path, args.dataset, origin_embedding, edge_index.T,
        args.prop_step, args.r, weights, args.num_threads,
        init_rmax, origin_embedding.shape[0]
    )
    del edge_index
    gc.collect()

    logger.info(f"Init Graph Prop cost: {time.perf_counter() - start:.6f}s")

    X = torch.FloatTensor(origin_embedding.T)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    del X

    # ═══════════════════════════════════════════════════
    # NEW: Store model kwargs for Gold Standard retraining
    # ═══════════════════════════════════════════════════
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
    evaluator = (
        Evaluator(name=args.dataset)
        if args.dataset in ["ogbn-arxiv", "ogbn-products"]
        else None
    )

    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model")
    checkpt_file = f"{args.analysis_path}/{args.dataset}/{name}_model/init_model.pt"
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

    # ═══════════════════════════════════════════════════
    # NEW: Save original model and compute base metrics
    # ═══════════════════════════════════════════════════
    original_checkpt = f"{args.analysis_path}/{args.dataset}/{name}_model/original_model.pt"
    torch.save(model.state_dict(), original_checkpt)

    if do_metrics:
        logger.info("--- Computing Base Model (Original) Metrics ---")
        base_metrics = compute_all_metrics(
            model_unlearned=model,
            device=device,
            X_test=X_test, y_test=y_test,
            X_train_remaining=X_train,
            y_train_remaining=y_train,
            batch_size=args.test_batch,
            model_retrained=None,
            X_forgotten=None,
            y_forgotten=None,
            mia_max_samples=mia_num_samples,
            compute_bounds=True,
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

    edge_file = f"{del_path}/{args.dataset}/{args.dataset}_del_edges{args.del_postfix}.npy"
    del_edges = np.load(edge_file)
    if del_edges.shape[1] == 2:
        del_edges = del_edges.T

    # ═══════════════════════════════════════════════════
    # NEW: Track cumulative forgotten nodes for MIA
    # ═══════════════════════════════════════════════════
    all_forgotten_nodes = set()

    # NEW: Precompute train node ID → position mapping (for MIA data prep)
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    train_idx_to_pos = {idx.item(): pos for pos, idx in enumerate(train_indices)}

    del X_train, X_val, X_test, noises
    gc.collect()

    logger.info("*" * 20)
    logger.info("--- Starting Unlearning (Edge Removal) Loop ---")
    start_time = time.perf_counter()
    edge_idx_start = args.edge_idx_start

    for i in range(args.num_batch_removes):
        edges = del_edges[
            :,
            edge_idx_start + i * args.num_removes:
            edge_idx_start + args.num_removes * (i + 1),
        ].T.tolist()

        # ═══════════════════════════════════════════════════
        # NEW: Track forgotten nodes (endpoints of deleted edges)
        # ═══════════════════════════════════════════════════
        for edge in edges:
            all_forgotten_nodes.update(edge)
            
        # ═══════════════════════════════════════════════════
        # 1. AGU FIX: Cache the training features BEFORE the C++ update
        # ═══════════════════════════════════════════════════
        X_old_train = torch.FloatTensor(origin_embedding.T)[train_mask]

        # ═══════════════════════════════════════════════════
        # FIX 3: UpdateEdges + UpdateFeatures
        # ═══════════════════════════════════════════════════
        struct_time = g.UpdateEdges(
            edges, origin_embedding, args.num_threads, update_rmax
        )

        affected_endpoints = list(set(node for edge in edges for node in edge))
        feat_time = g.UpdateFeatures(
            affected_endpoints, origin_embedding, args.num_threads, update_rmax
        )

        return_time = struct_time + feat_time
        update_cost.append(return_time)

        # Diagnostic for first 3 iterations
        if i < 3:
            logger.info(
                f"  struct_time={struct_time:.6f}, feat_time={feat_time:.6f}, "
                f"affected={len(affected_endpoints)}"
            )
            logger.info(f"  sample AFTER: {origin_embedding[0,:5]}")

        # ═══════════════════════════════════════════════════
        # 2. AGU FIX: Extract the new training features
        # ═══════════════════════════════════════════════════
        X_new = torch.FloatTensor(origin_embedding.T)
        update_finish_time = time.perf_counter()
        
        X_new_train = X_new[train_mask]
        X_new_val = X_new[val_mask]
        X_new_test = X_new[test_mask]
        del X_new
        
        # ═══════════════════════════════════════════════════
        # 3. AGU EXACT NEIGHBOR SELECTION (Experiment 3)
        # Compare old vs new features to find the EXACT affected subgraph
        # ═══════════════════════════════════════════════════
        feature_diffs = torch.norm(X_new_train - X_old_train, dim=1)
        # AGU Marginal Filtering: Only select nodes with meaningful feature shifts
        # 1e-6 is a safe threshold to ignore floating-point noise
        agu_affected_indices = (feature_diffs > 1e-6).nonzero(as_tuple=True)[0]
        del X_old_train
        
        
        model_params = [p for p in model.parameters() if p.requires_grad]
        new_grad = cal_grad_handloader(
            model, device, X_new_train, y_train, args.test_batch, retain=True
        )
        vs = tuple(
            (old_grad[k] - new_grad[k]).detach()
            for k in range(len(old_grad))
        )
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
                X_train=X_new_train, y_train=y_train,
                batch_size=lissa_bs,
                recursions=lissa_rec,
                damp=lissa_damp,
                scale=lissa_scale,
                convergence_tol=lissa_tol,
                check_every=lissa_check
            )

        apply_lissa_update(model, inverse_hvs, device,
                           max_update_ratio=max_update_r)
        apply_pgd_projection(model, pgd_c=pgd_c, mode=pgd_mode)

        if args.std > 0:
            noise_scale = args.std / math.sqrt(i + 2)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        per_param_grad = noise_scale / math.sqrt(p.numel())
                        p.data.add_(torch.randn_like(p) * per_param_grad)

            # FIX: Re-apply PGD after noise injection.
            # Noise can push individual parameter tensors outside the ||w||₂ ≤ C
            # ball required by CertDNN Theorem 3.4.  Without this projection the
            # PGD constraint is violated post-noise, invalidating the certification
            # guarantee.  This matches CertDNN Algorithm 1 which adds noise then
            # implicitly assumes the output is still in the feasible set.
            apply_pgd_projection(model, pgd_c=pgd_c, mode=pgd_mode)

        # ═══════════════════════════════════════════════════
        # 4. AGU FIX: Pass the EXACT affected nodes to NIM
        # ═══════════════════════════════════════════════════
        if do_nim:
            nim_finetune(
                model, device, X_new_train, y_train,
                agu_affected_indices, # <-- Replaced train_mask and edges
                lr=args.lr,
                nim_epochs=nim_epochs,
                pgd_c=pgd_c
            )
            
        del model_params, inverse_hvs, vs
        gc.collect()
        clear_cache(device)

        old_grad = cal_grad_handloader(
            model, device, X_new_train, y_train, args.test_batch, retain=False
        )

        # ═══════════════════════════════════════════════════
        # NEW: AGU standalone check (lightweight, no retrain needed)
        # ═══════════════════════════════════════════════════
        if do_agu and not do_metrics:
            _agu_wnorm = sum(
                p.data.norm(2).item()**2
                for p in model.parameters() if p.requires_grad
            ) ** 0.5
            _agu_gnorm = sum(g_tensor.norm().item()**2 for g_tensor in old_grad) ** 0.5
            _agu_pgd_ok = all(
                p.data.norm(2).item() <= pgd_c * (1 + 1e-6)
                for p in model.parameters() if p.requires_grad
            )
            logger.info(
                f"  [AGU] Batch {i}: ||grad||={_agu_gnorm:.6e}, "
                f"||w||={_agu_wnorm:.4f}, PGD_ok={_agu_pgd_ok}"
            )

        test_acc = test(model, device, X_new_test, y_test, args.test_batch, evaluator)
        val_acc = test(model, device, X_new_val, y_val, args.test_batch, evaluator)

        remove_finish_time = time.perf_counter()
        acc_removal[0].append(val_acc)
        acc_removal[1].append(test_acc)

        # ── Timing ──
        unlearn_time_i = remove_finish_time - update_finish_time
        total_time_i = unlearn_time_i + return_time
        unlearn_cost.append(unlearn_time_i)
        tot_cost.append(total_time_i)

        checkpt_file = (
            f"{args.analysis_path}/{args.dataset}/{name}_model/"
            f"Batch_{args.num_batch_removes}_Num_{args.num_removes}_"
            f"lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_"
            f"rmax_{origin_rmax}_std_{args.std}_prop_{args.prop_step}_"
            f"layer_{args.layer}_batch_{args.train_batch}_"
            f"edge_idx_{args.edge_idx_start}_{i}.pt"
        )
        torch.save(model.state_dict(), checkpt_file)

        # ═══════════════════════════════════════════════════
        # NEW: Full Research Metrics + CertDNN (when enabled)
        # ═══════════════════════════════════════════════════
        if do_metrics and (
            i % metric_every == 0 or i == args.num_batch_removes - 1
        ):
            logger.info(f"--- Computing Research Metrics for Batch {i} ---")

            # ── Prepare forgotten node features for MIA ──
            forgotten_positions = [
                train_idx_to_pos[n]
                for n in sorted(all_forgotten_nodes)
                if n in train_idx_to_pos
            ]

            X_forgotten = None
            y_forgotten = None
            if len(forgotten_positions) > 0:
                try:
                    X_forgotten = X_new_train[forgotten_positions]
                    y_forgotten = y_train[forgotten_positions]
                except Exception as e:
                    logger.warning(f"MIA data prep failed: {e}")
                    X_forgotten = None
                    y_forgotten = None

            # ── Optional: Retrain Gold Standard ──
            model_retrained = None
            retrain_time_i = float('nan')

            if do_retrain:
                retrain_checkpt = (
                    f"{args.analysis_path}/{args.dataset}/{name}_model/"
                    f"retrained_batch_{i}.pt"
                )

                retrain_noises = None
                if b_std > 0:
                    tmp_model = ClassMLP(**model_kwargs).to(device)
                    retrain_noises = [
                        b_std * torch.randn(p.shape).float().to(device)
                        for p in tmp_model.parameters() if p.requires_grad
                    ]
                    del tmp_model

                model_retrained, retrain_time_i = retrain_gold_standard(
                    model_class=ClassMLP,
                    model_kwargs=model_kwargs,
                    device=device,
                    X_train=X_new_train,
                    y_train=y_train,
                    X_val=X_new_val,
                    y_val=y_val,
                    batch_size=args.train_batch,
                    lr=args.lr,
                    wd=args.wd,
                    epochs=retrain_epochs,
                    patience=retrain_patience,
                    checkpt_file=retrain_checkpt,
                    evaluator=evaluator,
                    noises=retrain_noises,
                    pgd_c=pgd_c,
                )

                if save_retrained:
                    logger.info(f"Retrained model saved: {retrain_checkpt}")

                del retrain_noises
                gc.collect()
                clear_cache(device)

            # ── Compute standard metrics ──
            metrics = compute_all_metrics(
                model_unlearned=model,
                device=device,
                X_test=X_new_test,
                y_test=y_test,
                X_train_remaining=X_new_train,
                y_train_remaining=y_train,
                batch_size=args.test_batch,
                model_retrained=model_retrained,
                X_forgotten=X_forgotten,
                y_forgotten=y_forgotten,
                mia_max_samples=mia_num_samples,
                compute_bounds=True,
            )

            # ── CertDNN: Certified Unlearning Verification ──
            if do_certdnn:
                logger.info("  Running CertDNN Certification...")

                def deg_fn(node_id):
                    try:
                        return max(int(g.GetDegree(int(node_id))), 1)
                    except Exception:
                        return 1

                cert_metrics = compute_certification_bounds(
                    model_unlearned=model,
                    model_retrained=model_retrained,
                    device=device,
                    X_train=X_new_train,
                    y_train=y_train,
                    batch_size=args.test_batch,
                    edges_removed=edges,
                    deg_fn=deg_fn,
                    train_size=X_new_train.shape[0],
                    feat_dim=feat_dim,
                    lam=args.lam,
                    rmax=args.rmax,
                    num_nodes=origin_embedding.shape[1],
                    prop_step=args.prop_step,
                    pgd_c=pgd_c,
                    removal_type=cert_removal,
                )

                metrics.update(cert_metrics)
                log_certification_summary(cert_metrics, batch_idx=i, logger_ref=logger)

            # ── Add timing info ──
            metrics['batch_idx'] = i
            metrics['unlearn_time'] = unlearn_time_i
            metrics['update_time'] = return_time
            metrics['total_time'] = total_time_i
            metrics['retrain_time'] = retrain_time_i

            # ── Log & Save to CSV ──
            log_metrics_summary(metrics, batch_idx=i, logger_ref=logger)
            append_csv_row(csv_output, metrics)

            # ── Cleanup retrained model ──
            if model_retrained is not None:
                if not save_retrained:
                    try:
                        os.remove(retrain_checkpt)
                    except OSError:
                        pass
                del model_retrained
                gc.collect()
                clear_cache(device)

            if X_forgotten is not None:
                del X_forgotten, y_forgotten
                gc.collect()

        # ── Original cleanup ──
        del X_new_train, X_new_val, X_new_test

        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: Edge del = {edges[0]}, "
                f"Val acc = {acc_removal[0][-1]:.4f}, "
                f"Test acc = {acc_removal[1][-1]:.4f}, "
                f"avg update cost: {np.mean(update_cost[1:]):.4f}, "
                f"avg unlearn cost: {np.mean(unlearn_cost[1:]):.4f}, "
                f"avg tot cost: {np.mean(tot_cost[1:]):.4f}"
            )

    logger.info(f"Total Unlearning Process Cost: {time.perf_counter() - start_time:.4f}s")

    # ── Original file outputs (unchanged) ──
    np.savetxt(f_tot_cost, tot_cost, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")
    f_tot_cost.close()
    f_unlearn_cost.close()
    f_update_cost.close()
    f_acc.close()

    # ═══════════════════════════════════════════════════
    # NEW: Final summary when metrics are enabled
    # ═══════════════════════════════════════════════════
    if do_metrics:
        logger.info(f"{'='*60}")
        logger.info(f"  CSV Summary Sheet: {csv_output}")
        logger.info(f"  Total batches:     {args.num_batch_removes}")
        logger.info(f"  Forgotten nodes:   {len(all_forgotten_nodes)}")
        logger.info(f"  Final test acc:    {acc_removal[1][-1]:.4f}")
        logger.info(f"  Avg unlearn cost:  {np.mean(unlearn_cost[1:]):.4f}s")
        logger.info(f"  Avg total cost:    {np.mean(tot_cost[1:]):.4f}s")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()