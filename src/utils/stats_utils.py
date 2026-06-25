import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

_power_analysis = TTestPower()


def wilcoxon_battery(
    tests: dict,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    print_results: bool = True,
    label: str = None,
) -> pd.DataFrame:
    """Run multiple Wilcoxon signed-rank tests with multiple-comparison correction and power.

    Parameters
    ----------
    tests : dict
        Keys are test names. Values are either:
          - a 1-tuple/array for one-sample test against zero: (x,)
          - a 2-tuple for paired two-sample test: (x, y)
    alpha : float
        Significance level used for both correction and power calculation.
    correction : str
        Multiple-comparison method passed to statsmodels.multipletests.
    print_results : bool
        If True, print a summary table.
    label : str
        Optional header printed before results.

    Returns
    -------
    pd.DataFrame with columns: test, n, stat, p, corrected_p, cohen_d, power
    """
    names, stats_list, p_values, diffs, ns = [], [], [], [], []

    for name, args in tests.items():
        if isinstance(args, tuple) and len(args) == 2:
            x, y = args
            result = wilcoxon(x, y)
            d = np.asarray(x) - np.asarray(y)
        else:
            x = args[0] if isinstance(args, tuple) else args
            result = wilcoxon(x)
            d = np.asarray(x)

        names.append(name)
        stats_list.append(result.statistic)
        p_values.append(result.pvalue)
        diffs.append(d)
        ns.append(len(d))

    corrected_p = multipletests(p_values, method=correction)[1]

    rows = []
    for name, stat, p, corr_p, d, n in zip(names, stats_list, p_values, corrected_p, diffs, ns):
        cohen_d = d.mean() / d.std() if d.std() > 0 else np.nan
        power = _power_analysis.solve_power(effect_size=cohen_d, nobs=n, alpha=alpha) if not np.isnan(cohen_d) else np.nan
        rows.append({"test": name, "n": n, "stat": stat, "p": p, "corrected_p": corr_p, "cohen_d": cohen_d, "power": power})

    df = pd.DataFrame(rows)

    if print_results:
        if label:
            print(f"Group: {label}")
        for _, row in df.iterrows():
            print(f"  {row['test']}: p = {row['p']:.3f}, corrected p = {row['corrected_p']:.3f}, cohen_d = {row['cohen_d']:.2f}, power = {row['power']:.2f}  (n={int(row['n'])})")
        print()

    return df
