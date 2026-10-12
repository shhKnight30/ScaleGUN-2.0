import argparse

def argparser():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument("--dataset", default="ogbn-arxiv", type=str) # CHANGED: ogbn-arxiv
    parser.add_argument("--path", default="./data/")
    parser.add_argument("--del_path_suffix", default="unlearning_data/")
    parser.add_argument("--analysis_path", default="./analysis")
    parser.add_argument("--trials", type=int, default=5) # CHANGED: 5 trials

    # args for preprocessing
    parser.add_argument("--axis_num", default=1, type=int, choices=[1, 0])

    # args for propagation
    parser.add_argument("--prop_algo", type=str,
                        choices=["power", "push", "MC"], default="push")
    parser.add_argument("--prop_step", default=2, type=int) # CHANGED: 2-hop propagation
    parser.add_argument("--r", default=0.5, type=float)
    parser.add_argument("--decay", default=0.1, type=float)
    parser.add_argument("--RW", type=int, default=500,
                        help="random walk times")
    parser.add_argument("--rmax", default=1e-10, type=float) # CHANGED: Guaranteed 0 retrains
    parser.add_argument("--ppr", default=False, action="store_true")
    parser.add_argument("--weight_mode", default="test",
                        type=str, choices=["decay", "avg", "test", "hetero"],)
    parser.add_argument("--num_threads", default=40, type=int)

    # args for model, loss function
    parser.add_argument("--layer", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hidden", default=1024, type=int)
    # Set delta<0 here, then delta = 1/|E| will be set in the edge experiments, and delta = 1/|V| in the node experiments.
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1.0,
                        help="Eps coefficient for certified removal.")
    parser.add_argument("--lam", type=float, default=5e-4, # CHANGED: Deep model regularization (ScaleGUN Table 8)
                        help="L2 regularization")
    parser.add_argument("--std", type=float, default=1e-1,
                        help="standard deviation for objective perturbation",)
    parser.add_argument("--noise_mode", type=str, default="data",
                        help="Data dependent noise or worst case noise [data/worst].",)

    # args for training
    parser.add_argument("--lr", default=1e-3, type=float) # CHANGED: Deep model learning rate (ScaleGUN Table 8)
    parser.add_argument("--dropout", type=float,
                        default=0.3, help="dropout rate.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of optimization steps")
    parser.add_argument("--train_mode", type=str,
                        default="ovr", help="train mode [ovr/binary]")
    parser.add_argument("--Y_binary", type=str, default="0",
                        help="In binary mode, is Y_binary class or Y_binary_1 vs Y_binary_2 (i.e., 0+1).",)
    parser.add_argument("--init_method", type=str, default="kaiming",
                        help="init method for parameter w [kaiming/xavier/zero]",)
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Choice of optimizer. [LBFGS/Adam]",)
    parser.add_argument("--wd", type=float, default=5e-4,
                        help="Weight decay factor for Adam")
    parser.add_argument("--dev", default=1, type=int,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--train_batch", default=1024, type=int) # CHANGED: Deep model batch size (ScaleGUN Table 8)
    parser.add_argument("--test_batch", default=100000, type=int)
    parser.add_argument("--checkid", default=0, type=int)

    # args for log
    parser.add_argument("--verbose", action="store_true",
                        default=False, help="verbosity in optimizer")
    parser.add_argument("--disp", type=int, default=10)

    # args for unlearing
    parser.add_argument("--compare_gnorm", action="store_true", default=False,
                        help="Compute norm of worst case and real gradient each round.",)
    parser.add_argument("--no_retrain", action="store_true", default=False)
    parser.add_argument("--num_batch_removes", default=5, type=int)
    parser.add_argument("--edge_idx_start", default=0, type=int)
    parser.add_argument("--damping", default=1e-4, type=float)
    parser.add_argument("--num_removes", default=1, type=int,
                        help="number of removed edges/nodes in each batch",)
    parser.add_argument("--removal_mode", default="node", type=str)
    parser.add_argument("--compare_retrain",
                        action="store_true", default=False)
    parser.add_argument('--compute_metrics', action='store_true', default=False)
    parser.add_argument('--retrain', action='store_true', default=False)
    parser.add_argument('--agu_check', action='store_true', default=False)
    
    parser.add_argument('--csv_output', type=str, default='')
    parser.add_argument('--metric_every', type=int, default=1)
    parser.add_argument('--mia_num_samples', type=int, default=500)
    
    # ─────────────────────────────────────────────────────────────────────────
    # pgd_c: PGD parameter-norm bound (CertDNN, Zhang et al. 2024, ICML).
    #
    # The CertDNN paper constrains ||w||₂ ≤ C so that the approximation error
    # bound ||w̃ - w̃*||₂ ≤ 2C(MC + λ)/(λ + λmin) remains tractable (Theorem 3.4).
    # They set C close to the *natural weight norm* of the trained model so that
    # PGD rarely activates during training, preserving utility (Appendix E.1).
    #   MLP / MNIST    → C = 10  (global norm, small model)
    #   AllCNN / CIFAR → C = 20  (global norm)
    #   ResNet / SVHN  → C = 20  (global norm)
    #
    # Our code uses per_param PGD: each layer tensor is bounded independently.
    # For ClassMLP with hidden=1024 on ogbn-arxiv, Kaiming-init per-layer norms
    # (empirical approximation):
    #   Input  (128→1024):   ||W||₂ ≈ √(128×1024) × 0.125 ≈ 45
    #   Hidden (1024→1024):  ||W||₂ ≈ √(1024²)    × 0.044 ≈ 45
    #   Output (1024→40):    ||W||₂ ≈ √(1024×40)  × 0.044 ≈  9
    # With L2 weight-decay (wd=5e-4), trained norms fall to roughly 25–45.
    # Setting C=50 ensures PGD does not clip during normal training (no utility
    # loss) while providing a tighter bound than C=100 (which rarely activates
    # and makes the CertDNN bound vacuous).
    #
    # Tune: if test accuracy drops >1% vs baseline, increase C by 10.
    #        if certification ratio > 2.0, decrease C by 10.
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument('--pgd_c', type=float, default=50.0,
                        help='Per-parameter L2 norm bound for PGD projection. '
                             'Set near the natural weight norm of the trained '
                             'model so PGD does not clip during training. '
                             'For ClassMLP hidden=1024: ~50. See CertDNN '
                             '(Zhang et al., ICML 2024) Appendix E.1.')
    parser.add_argument('--pgd_mode', type=str, default='per_param')
    
    parser.add_argument('--nim_epochs', type=int, default=2) # CHANGED: Faster NIM fine-tuning
    
    parser.add_argument('--retrain_epochs', type=int, default=100)
    parser.add_argument('--retrain_patience', type=int, default=30)
    parser.add_argument('--save_retrained_model', action='store_true', default=False)
    
    parser.add_argument('--cert_removal_type', type=str, default='edge')
    
    parser.add_argument('--lissa_batch_size', type=int, default=256)
    # ─────────────────────────────────────────────────────────────────────────
    # lissa_recursions: number of LiSSA iterations for inverse-Hessian approx.
    #
    # From CertDNN (Zhang et al., ICML 2024), Appendix D:
    #   "number of recursion s: 1,000" for all architectures (MLP, AllCNN,
    #   ResNet18) — without early stopping.
    #
    # Theorem 3.6 requires s ≥ 2(L+λ)/(λ+λmin) × ln((L+λ)/(λ+λmin)).
    # For a deep MLP with hidden=1024, L (Lipschitz gradient constant) is
    # large (O(||W||² per layer), so convergence requires more iterations than
    # shallow models.  With our convergence_tol=1e-4 early-stopping check
    # every 50 iterations, we typically exit at 100–400 iterations in practice.
    # Setting max=500 gives the algorithm room to converge on difficult batches
    # while remaining ≈2× cheaper than the paper's 1,000 (which had no early
    # stopping).  Increase to 1000 if lissa_inverse_hvp logs show divergence
    # or ||H⁻¹v|| is unstable across batches.
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument('--lissa_recursions', type=int, default=500,
                        help='Max LiSSA iterations for inverse-HVP. '
                             'CertDNN (Zhang et al., ICML 2024) uses 1000 '
                             'without early stopping; 500 with convergence_tol '
                             'early-stopping is empirically sufficient for '
                             'ClassMLP hidden=1024.')
    parser.add_argument('--lissa_damp', type=float, default=0.5)
    parser.add_argument('--lissa_scale', type=float, default=50.0)
    parser.add_argument('--lissa_convergence_tol', type=float, default=1e-4)
    parser.add_argument('--lissa_check_every', type=int, default=50)
    parser.add_argument('--max_update_ratio', type=float, default=0.01)

    # args for attack
    parser.add_argument("--num_add", type=int, default=10000)
    parser.add_argument("--max_deg", type=int, default=5)
    parser.add_argument("--del_postfix", type=str, default="")

    parser.add_argument("--test_MI", action="store_true", default=False)
    parser.add_argument("--replay", action="store_true", default=False)
    parser.add_argument("--attack_dim", type=int, default=100)
    parser.add_argument("--rand", action="store_true")
    
    # for Node influence Maximization ::
    parser.add_argument('--use_nim', action='store_true', help='Enable Experiment 2: NIM fine-tuning' , default=False)

    # --- [EXPERIMENT 4 FLAG] ---
    parser.add_argument('--use_certdnn', action='store_true', default=False,
                        help='Enable CertDNN PGD constraint and constant error bound (Exp 4)')
    
    args = parser.parse_args()

    if args.eps < 0:
        args.eps = 0.1/args.std
    return args