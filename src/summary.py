"""Print LaTeX tables from models/metrics.json (nothing is hard-coded).

Run from the repo root:  python -m src.summary
"""
from src.predictor import load_metrics


def latex_table(header, rows, caption, label):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    metrics = load_metrics()
    data = metrics["dataset"]

    cv_rows = [
        [
            name,
            f"{m['r2_mean']:.4f} $\\pm$ {m['r2_std']:.4f}",
            f"{m['mae_mean']:.2f} $\\pm$ {m['mae_std']:.2f}",
        ]
        for name, m in metrics["cv"].items()
    ]
    print(latex_table(
        ["Algorithm", "R-squared", "MAE (INR)"],
        cv_rows,
        f"5-fold cross-validation on the training set ({data['train_rows']} rows)",
        "tab:cv",
    ))
    print()

    labels = {
        "full": "All test rows",
        "in_range": f"Price $\\leq$ {data['price_iqr_high']:.0f} INR",
    }
    test_rows = [
        [labels[key], str(m["n"]), f"{m['r2']:.4f}", f"{m['mae']:.2f}", f"{m['rmse']:.2f}"]
        for key, m in metrics["test"].items()
    ]
    print(latex_table(
        ["Test subset", "Rows", "R-squared", "MAE (INR)", "RMSE (INR)"],
        test_rows,
        f"Hold-out test performance of {metrics['best_model']}",
        "tab:results",
    ))


if __name__ == "__main__":
    main()
