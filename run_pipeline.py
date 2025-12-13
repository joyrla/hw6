"""
Serverless Cold-Start Prediction Pipeline

This script predicts serverless function invocation traffic using:
1. Naive Baseline (Lag-1 prediction)
2. AI Pipeline (Chronos pre-trained time series model)

Usage: python run_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
np.random.seed(42)

# Find data - check both relative and parent paths
DATA_CANDIDATES = [
    "data/azure-functions-dataset-2019/invocations_per_function_md.feather",
    "../data/azure-functions-dataset-2019/invocations_per_function_md.feather",
]

def find_data_path():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    return None

def main():
    print("=" * 60)
    print("SERVERLESS COLD-START PREDICTION PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    data_path = find_data_path()
    if not data_path:
        print("\nError: Data file not found!")
        print("Please download Azure Functions dataset and place in data/ folder")
        print("Expected: data/azure-functions-dataset-2019/invocations_per_function_md.feather")
        return
    
    print(f"\n[1/5] Loading data from {data_path}...")
    df = pd.read_feather(data_path)
    print(f"      Dataset: {df.shape[0]:,} functions, {df.shape[1]} columns")
    
    # 2. Select target function
    print("\n[2/5] Preprocessing...")
    metadata_cols = ['HashOwner', 'HashApp', 'HashFunction', 'Trigger']
    time_cols = [col for col in df.columns if col not in metadata_cols]
    
    df['total_traffic'] = df[time_cols].sum(axis=1)
    df['traffic_variance'] = df[time_cols].var(axis=1)
    df['selection_score'] = df['total_traffic'] * np.log1p(df['traffic_variance'])
    
    target_idx = df['selection_score'].idxmax()
    target_func = df.loc[target_idx]
    traffic = target_func[time_cols].values.astype(np.float32)
    
    split_idx = int(len(traffic) * 0.8)
    print(f"      Selected function with {target_func['total_traffic']:,.0f} invocations")
    print(f"      Train: {split_idx}, Test: {len(traffic) - split_idx}")
    
    # 3. Naive Baseline
    print("\n[3/5] Running Naive Baseline (Lag-1)...")
    baseline_preds = traffic[split_idx-1:-1]
    actuals = traffic[split_idx:]
    
    baseline_mae = np.mean(np.abs(actuals - baseline_preds))
    baseline_cold_starts = int(np.sum(baseline_preds < actuals))
    print(f"      Baseline MAE: {baseline_mae:.2f}")
    
    # 4. AI Pipeline (Chronos)
    print("\n[4/5] Running AI Pipeline (Chronos)...")
    try:
        import torch
        from chronos import ChronosPipeline
        
        # Set torch seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"      Device: {device}")
        
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=device,
            dtype=torch.float32,
        )
        
        ai_preds = []
        context_length = 512
        
        print("      Generating predictions...")
        for i, test_idx in enumerate(range(split_idx, len(traffic))):
            start_idx = max(0, test_idx - context_length)
            context = traffic[start_idx:test_idx]
            context_tensor = torch.tensor(context, dtype=torch.float32)
            
            forecast = pipeline.predict(context_tensor, prediction_length=1, num_samples=20)
            pred = float(forecast.median(dim=1).values.flatten()[0])
            ai_preds.append(max(0.0, pred))
            
            if (i + 1) % 50 == 0:
                print(f"      Processed {i + 1}/{len(actuals)}...")
        
        ai_preds = np.array(ai_preds, dtype=np.float32)
        ai_mae = np.mean(np.abs(actuals - ai_preds))
        ai_cold_starts = int(np.sum(ai_preds < actuals))
        
    except ImportError as e:
        print(f"      Error: {e}")
        print("      Install with: pip install chronos-forecasting torch")
        ai_mae, ai_cold_starts = 0, 0
        ai_preds = np.zeros_like(actuals)
    
    # 5. Results
    print("\n[5/5] Results Summary")
    print("=" * 60)
    print(f"{'Method':<20} | {'MAE':<12} | {'Cold Starts':<12}")
    print("-" * 60)
    print(f"{'Baseline (Lag-1)':<20} | {baseline_mae:<12.2f} | {baseline_cold_starts:<12}")
    print(f"{'AI (Chronos)':<20} | {ai_mae:<12.2f} | {ai_cold_starts:<12}")
    print("=" * 60)
    
    if ai_mae > 0:
        improvement = 100 * (baseline_mae - ai_mae) / baseline_mae
        print(f"\nMAE Improvement: {improvement:.1f}%")
    
    # Calculate errors for case studies
    baseline_errors = actuals - baseline_preds
    ai_errors = actuals - ai_preds
    
    # Save figures
    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: Predictions comparison (same as notebook)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(actuals))
    ax.plot(x, actuals, 'k-', label='Actual', alpha=0.7)
    ax.plot(x, baseline_preds, 'b--', label='Baseline (Lag-1)', alpha=0.6)
    ax.plot(x, ai_preds, 'r--', label='AI (Chronos)', alpha=0.6)
    ax.set_xlabel('Test Index (minutes)')
    ax.set_ylabel('Invocations')
    ax.set_title('Predictions vs Actual Traffic')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/predictions_comparison.png', dpi=150)
    plt.close()
    
    # Figure 2: Case studies (3-panel, same as notebook)
    abs_diff = np.abs(ai_preds - baseline_preds)
    ai_better_mask = np.abs(ai_errors) < np.abs(baseline_errors)
    ai_better_indices = np.where(ai_better_mask)[0]
    ex1_idx = ai_better_indices[np.argmax(abs_diff[ai_better_mask])] if len(ai_better_indices) > 0 else 0
    baseline_better_mask = np.abs(baseline_errors) < np.abs(ai_errors)
    baseline_better_indices = np.where(baseline_better_mask)[0]
    ex2_idx = baseline_better_indices[np.argmax(abs_diff[baseline_better_mask])] if len(baseline_better_indices) > 0 else len(actuals)//2
    ex3_idx = np.argmax(actuals)
    
    def plot_case_study(ax, center_idx, title, annotation, annotation_pos):
        window = 8
        start = max(0, center_idx - window)
        end = min(len(actuals), center_idx + window + 1)
        x = np.arange(start, end)
        ax.plot(x, actuals[start:end], 'ko-', linewidth=2, markersize=8, label='Actual Traffic')
        ax.plot(x, baseline_preds[start:end], 'b^--', linewidth=1.5, markersize=7, label='Baseline (Lag-1)')
        ax.plot(x, ai_preds[start:end], 'rs--', linewidth=1.5, markersize=7, label='AI (Chronos)')
        ax.axvline(x=center_idx, color='green', linestyle=':', alpha=0.7, linewidth=2)
        base_pct = 100 * abs(baseline_errors[center_idx]) / actuals[center_idx]
        ai_pct = 100 * abs(ai_errors[center_idx]) / actuals[center_idx]
        ax.annotate(f'{annotation}\n(AI: {ai_pct:.1f}% vs Base: {base_pct:.1f}%)',
                    xy=(center_idx, ai_preds[center_idx]), xytext=annotation_pos,
                    fontsize=9, ha='left', arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
        ax.set_xlabel('Test Index (minutes)', fontsize=10)
        ax.set_ylabel('Invocations', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_case_study(axes[0], ex1_idx, f'Case 1: AI Outperforms (Index {ex1_idx})',
                    'AI anticipated\nrecovery', (ex1_idx + 2, ai_preds[ex1_idx] + 10000))
    plot_case_study(axes[1], ex2_idx, f'Case 2: Baseline Outperforms (Index {ex2_idx})',
                    'Baseline closer\nduring decline', (ex2_idx + 2, baseline_preds[ex2_idx] + 10000))
    plot_case_study(axes[2], ex3_idx, f'Case 3: Peak Traffic (Index {ex3_idx})',
                    'Both handle\nsteady traffic', (ex3_idx + 2, actuals[ex3_idx] - 5000))
    plt.tight_layout()
    plt.savefig('figures/case_studies.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Figures saved:")
    print("   - figures/predictions_comparison.png")
    print("   - figures/case_studies.png")

if __name__ == "__main__":
    main()
