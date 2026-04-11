"""
Generate example screenshots showing what Evidently reports look like.

Run this to create sample visualizations for GitHub documentation.
Actual reports are interactive HTML with more features.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
plt.style.use('seaborn-v0_8-darkgrid')

def create_data_drift_report():
    """Create example data drift report screenshot."""
    fig = plt.figure(figsize=(14, 10))

    # Title
    fig.suptitle('Evidently Data Drift Report (Example)', fontsize=18, fontweight='bold', y=0.98)

    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[0.4, 1, 1], hspace=0.3, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.05)

    # ─────────────────────────────────────────────────────────────
    # 1. Summary Table (top, spanning both columns)
    # ─────────────────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')

    drift_data = [
        ['Feature', 'Drift Score (PSI)', 'Status'],
        ['transaction_amount', '0.74', '🔴 DRIFTED'],
        ['credit_limit', '0.45', '🔴 DRIFTED'],
        ['distance_from_home', '0.28', '🟡 WARNING'],
        ['merchant_category_code', '0.22', '🟡 WARNING'],
        ['velocity_score', '0.08', '🟢 STABLE'],
        ['transaction_hour', '0.06', '🟢 STABLE'],
    ]

    table = ax_table.table(cellText=drift_data, cellLoc='left', loc='center',
                           colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Header styling
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(weight='bold', color='white')

    # Row styling based on status
    for i in range(1, len(drift_data)):
        status = drift_data[i][2]
        if 'DRIFTED' in status:
            color = '#FFE4E1'
        elif 'WARNING' in status:
            color = '#FFF8DC'
        else:
            color = '#E8F5E9'

        for j in range(3):
            table[(i, j)].set_facecolor(color)

    ax_table.text(0.5, 1.15, 'Drift Summary: 2 features drifted, 2 warnings, 2 stable',
                  transform=ax_table.transAxes, ha='center', fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ─────────────────────────────────────────────────────────────
    # 2. Feature Distribution: transaction_amount (drifted)
    # ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])

    # Reference (training) distribution
    ref_data = np.random.lognormal(mean=4.5, sigma=1.2, size=5000)
    ref_data = np.clip(ref_data, 0, 500)

    # Current (inference) distribution - shifted
    curr_data = np.random.lognormal(mean=5.0, sigma=1.3, size=5000)
    curr_data = np.clip(curr_data, 0, 500)

    bins = np.linspace(0, 500, 30)
    ax1.hist(ref_data, bins=bins, alpha=0.5, label='Reference (Training)',
             color='#4A90E2', edgecolor='black', linewidth=0.5)
    ax1.hist(curr_data, bins=bins, alpha=0.5, label='Current (Inference)',
             color='#E57373', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Transaction Amount ($)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('transaction_amount (PSI: 0.74 - DRIFTED)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────
    # 3. Feature Distribution: credit_limit (drifted)
    # ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])

    # Reference distribution
    ref_data2 = np.random.normal(loc=5000, scale=2000, size=5000)
    ref_data2 = np.clip(ref_data2, 1000, 15000)

    # Current distribution - shifted right
    curr_data2 = np.random.normal(loc=7000, scale=2200, size=5000)
    curr_data2 = np.clip(curr_data2, 1000, 15000)

    bins2 = np.linspace(1000, 15000, 30)
    ax2.hist(ref_data2, bins=bins2, alpha=0.5, label='Reference (Training)',
             color='#4A90E2', edgecolor='black', linewidth=0.5)
    ax2.hist(curr_data2, bins=bins2, alpha=0.5, label='Current (Inference)',
             color='#E57373', edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Credit Limit ($)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('credit_limit (PSI: 0.45 - DRIFTED)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────
    # 4. Feature Distribution: velocity_score (stable)
    # ─────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])

    # Both distributions similar
    ref_data3 = np.random.normal(loc=0.5, scale=0.15, size=5000)
    ref_data3 = np.clip(ref_data3, 0, 1)

    curr_data3 = np.random.normal(loc=0.52, scale=0.16, size=5000)
    curr_data3 = np.clip(curr_data3, 0, 1)

    bins3 = np.linspace(0, 1, 30)
    ax3.hist(ref_data3, bins=bins3, alpha=0.5, label='Reference (Training)',
             color='#4A90E2', edgecolor='black', linewidth=0.5)
    ax3.hist(curr_data3, bins=bins3, alpha=0.5, label='Current (Inference)',
             color='#81C784', edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('Velocity Score', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('velocity_score (PSI: 0.08 - STABLE)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────
    # 5. PSI Trend Over Time
    # ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])

    days = np.arange(1, 15)
    psi_transaction = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.48, 0.62, 0.68, 0.72, 0.74, 0.75, 0.76, 0.74])
    psi_credit = np.array([0.04, 0.06, 0.10, 0.15, 0.22, 0.28, 0.35, 0.38, 0.42, 0.44, 0.45, 0.46, 0.47, 0.45])
    psi_velocity = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.09, 0.08, 0.07, 0.08, 0.09, 0.08, 0.08])

    ax4.plot(days, psi_transaction, marker='o', linewidth=2, label='transaction_amount', color='#E57373')
    ax4.plot(days, psi_credit, marker='s', linewidth=2, label='credit_limit', color='#FF9800')
    ax4.plot(days, psi_velocity, marker='^', linewidth=2, label='velocity_score', color='#81C784')

    # Threshold lines
    ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Minor drift (0.1)')
    ax4.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Major drift (0.25)')

    ax4.set_xlabel('Days Since Deployment', fontsize=11)
    ax4.set_ylabel('PSI Score', fontsize=11)
    ax4.set_title('PSI Drift Trend Over Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(alpha=0.3)

    # Footer note
    fig.text(0.5, 0.01, 'This is a static example. Run notebooks/2a_inference_monitoring.ipynb for interactive Evidently HTML reports.',
             ha='center', fontsize=10, style='italic', color='gray')

    output_path = OUTPUT_DIR / 'data-drift-report-example.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {output_path.name}")

def create_classification_performance_report():
    """Create example classification performance report screenshot."""
    fig = plt.figure(figsize=(14, 10))

    fig.suptitle('Evidently Classification Performance Report (Example)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35,
                          left=0.08, right=0.95, top=0.92, bottom=0.05)

    # ─────────────────────────────────────────────────────────────
    # 1. Metrics Table
    # ─────────────────────────────────────────────────────────────
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.axis('off')

    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Accuracy', '0.9245', '🟢 Good'],
        ['Precision', '0.8876', '🟢 Good'],
        ['Recall', '0.9123', '🟢 Good'],
        ['F1-Score', '0.8998', '🟢 Good'],
        ['ROC-AUC', '0.9567', '🟢 Good'],
    ]

    table = ax_metrics.table(cellText=metrics_data, cellLoc='center', loc='center',
                            colWidths=[0.3, 0.3, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(weight='bold', color='white')

    for i in range(1, len(metrics_data)):
        for j in range(3):
            table[(i, j)].set_facecolor('#E8F5E9')

    # ─────────────────────────────────────────────────────────────
    # 2. Confusion Matrix
    # ─────────────────────────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[1, 0])

    # Confusion matrix values
    cm = np.array([[8620, 180],
                   [380, 4820]])

    im = ax_cm.imshow(cm, cmap='Blues', alpha=0.6)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax_cm.text(j, i, str(cm[i, j]),
                            ha="center", va="center", color="black",
                            fontsize=16, fontweight='bold')

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Non-Fraud\n(Predicted)', 'Fraud\n(Predicted)'], fontsize=10)
    ax_cm.set_yticklabels(['Non-Fraud\n(Actual)', 'Fraud\n(Actual)'], fontsize=10)
    ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=10)

    # Add labels
    ax_cm.text(0, -0.5, 'TN: 8620', ha='center', fontsize=9, color='green', weight='bold',
               transform=ax_cm.transData)
    ax_cm.text(1, -0.5, 'FP: 180', ha='center', fontsize=9, color='red', weight='bold',
               transform=ax_cm.transData)
    ax_cm.text(0, 1.5, 'FN: 380', ha='center', fontsize=9, color='red', weight='bold',
               transform=ax_cm.transData)
    ax_cm.text(1, 1.5, 'TP: 4820', ha='center', fontsize=9, color='green', weight='bold',
               transform=ax_cm.transData)

    # ─────────────────────────────────────────────────────────────
    # 3. ROC Curve
    # ─────────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[1, 1])

    # Simulated ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr) ** 2.5  # Curve shape

    ax_roc.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = 0.96)', color='#4A90E2')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Random Classifier')

    ax_roc.set_xlabel('False Positive Rate', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate', fontsize=11)
    ax_roc.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(alpha=0.3)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])

    # ─────────────────────────────────────────────────────────────
    # 4. Precision-Recall Curve
    # ─────────────────────────────────────────────────────────────
    ax_pr = fig.add_subplot(gs[1, 2])

    recall = np.linspace(0, 1, 100)
    precision = 0.95 - 0.4 * recall ** 2

    ax_pr.plot(recall, precision, linewidth=3, label='PR Curve', color='#66BB6A')
    ax_pr.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='Baseline')

    ax_pr.set_xlabel('Recall', fontsize=11)
    ax_pr.set_ylabel('Precision', fontsize=11)
    ax_pr.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax_pr.legend(loc='upper right')
    ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1])

    # ─────────────────────────────────────────────────────────────
    # 5. Prediction Distribution
    # ─────────────────────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[2, :2])

    # Fraud probabilities for non-fraud (should be low)
    non_fraud_probs = np.random.beta(2, 8, 8800)

    # Fraud probabilities for fraud (should be high)
    fraud_probs = np.random.beta(8, 2, 5200)

    bins = np.linspace(0, 1, 50)
    ax_dist.hist(non_fraud_probs, bins=bins, alpha=0.6, label='Non-Fraud (Actual)',
                 color='#4A90E2', edgecolor='black', linewidth=0.5)
    ax_dist.hist(fraud_probs, bins=bins, alpha=0.6, label='Fraud (Actual)',
                 color='#E57373', edgecolor='black', linewidth=0.5)

    # Decision threshold line
    ax_dist.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                    label='Decision Threshold (0.5)')

    ax_dist.set_xlabel('Predicted Fraud Probability', fontsize=11)
    ax_dist.set_ylabel('Count', fontsize=11)
    ax_dist.set_title('Prediction Distribution by Actual Class', fontsize=12, fontweight='bold')
    ax_dist.legend(loc='upper center')
    ax_dist.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────
    # 6. Performance Over Time
    # ─────────────────────────────────────────────────────────────
    ax_time = fig.add_subplot(gs[2, 2])

    days = np.arange(1, 15)
    accuracy = 0.92 + 0.005 * np.random.randn(14)
    f1 = 0.90 + 0.008 * np.random.randn(14)

    ax_time.plot(days, accuracy, marker='o', linewidth=2, label='Accuracy', color='#4A90E2')
    ax_time.plot(days, f1, marker='s', linewidth=2, label='F1-Score', color='#66BB6A')
    ax_time.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Alert Threshold')

    ax_time.set_xlabel('Days Since Deployment', fontsize=11)
    ax_time.set_ylabel('Score', fontsize=11)
    ax_time.set_title('Performance Over Time', fontsize=12, fontweight='bold')
    ax_time.legend(loc='lower left', fontsize=9)
    ax_time.grid(alpha=0.3)
    ax_time.set_ylim([0.7, 1.0])

    # Footer note
    fig.text(0.5, 0.01, 'This is a static example. Run notebooks/2a_inference_monitoring.ipynb for interactive Evidently HTML reports.',
             ha='center', fontsize=10, style='italic', color='gray')

    output_path = OUTPUT_DIR / 'classification-performance-example.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {output_path.name}")

if __name__ == "__main__":
    print("📊 Generating Evidently report example screenshots...")
    print()
    create_data_drift_report()
    create_classification_performance_report()
    print()
    print("✅ Screenshots generated in docs/screenshots/evidently/")
    print("   These are example visualizations. Actual Evidently reports are interactive HTML.")
