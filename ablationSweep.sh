for w in 0.0 0.01 0.02 0.05 0.1 0.2; do
    python optimize_sequential.py --anchor-weight $w --output-dir outputs/ablation_anchor_${w}
done
