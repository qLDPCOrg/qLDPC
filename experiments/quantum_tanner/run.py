#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import run_randomized_search as search

import qldpc.cache

file_dir = os.path.dirname(__file__)
cache = qldpc.cache.get_disk_cache(".code_cache", cache_dir=file_dir)


columns = ["group", "base_code", "sample", "num_qubits", "dimension", "distance", "weight", "ratio"]
rows = []

for order, index, base_code in cache.iterkeys():
    if not cache[order, index, base_code]:
        cache.delete((order, index, base_code))
        continue

    for sample, (nn, kk, dd, num_trials, ww) in enumerate(cache[order, index, base_code]):
        if num_trials != search.NUM_TRIALS:
            continue
        ratio = kk * dd**2 / nn
        rows.append([(order, index), base_code, sample, nn, kk, dd, ww, ratio])

# collect all data into one data frame
data = pandas.DataFrame(rows, columns=columns)


####################################################################################################


def plot_logs(
    data: pandas.DataFrame,
    *,
    alpha: float = 0.1,
    refline: bool = True,
    figsize: tuple[int, int] = (4, 3),
):
    fig, ax = plt.subplots(figsize=figsize)
    log_n = np.log(data["num_qubits"].values)
    log_k = np.log(data["dimension"].values)
    log_d = np.log(data["distance"].values)
    ax.scatter(log_d / log_n, log_k / log_n, marker=".", color="k", alpha=alpha)

    if refline:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([0, 0.5], [1, 0], "r", label="$kd^2=n$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax.legend(loc="best")
    ax.set_xlabel(r"$\log_n(d_{\mathrm{upper}})$")
    ax.set_ylabel(r"$\log_n(k)$")
    fig.tight_layout()
    return fig, ax


def plot_rates(
    data: pandas.DataFrame,
    *,
    alpha: float = 0.1,
    refline: bool = True,
    figsize: tuple[int, int] = (4, 3),
):
    fig, ax = plt.subplots(figsize=figsize)
    dd = data["distance"].values
    kk_nn = data["dimension"].values / data["num_qubits"].values
    ax.scatter(dd, kk_nn, marker=".", color="k", alpha=alpha)

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)

    if refline:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        dd_vals = np.logspace(*map(np.log10, xlim), 200)
        with np.errstate(divide="ignore"):
            kk_nn_vals = 1 / dd_vals**2
        ax.plot(dd_vals, kk_nn_vals, "r", label="$kd^2=n$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax.legend(loc="best")
    ax.set_xlabel(r"distance bound ($d_{\mathrm{upper}}$)")
    ax.set_ylabel(r"code rate ($k/n$)")
    fig.tight_layout()
    return fig, ax


def plot_ratios(
    data: pandas.DataFrame,
    *,
    alpha: float = 0.1,
    refline: bool = True,
    figsize: tuple[int, int] = (4, 3),
):
    fig, ax = plt.subplots(figsize=figsize)
    dd = data["distance"].values
    rr = data["ratio"].values
    ax.scatter(dd, rr, marker=".", color="k", alpha=alpha)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if refline:
        xlim = ax.get_xlim()
        ax.plot(xlim, [1, 1], "r", label="$kd^2=n$")
        ax.set_xlim(*xlim)

    ax.legend(loc="best")
    ax.set_xlabel(r"$d_{\mathrm{upper}}$")
    ax.set_ylabel(r"$kd^2/n$")
    fig.tight_layout()
    return fig, ax


####################################################################################################

# selection = slice(None)
# # selection = (data["distance"] > 5) & (data["dimension"] / data["num_qubits"] > 0.1)
# plot_logs(data[selection])
# plt.show()

# data = data[data["num_qubits"] <= 98]

# params = set()
# for index, code in data.iterrows():
#     params.add((code["num_qubits"], code["dimension"], code["distance"], code["weight"]))
# for code in sorted(params):
#     print(code)
# exit()

fig, ax = plot_rates(data)
ax.set_title("random quantum Tanner codes")
fig.tight_layout()
plt.show()

exit()

data = data[data["distance"] >= 5]
data = data[data["dimension"] >= 4]

base_code = "Mittal-6"

for group in search.get_small_groups():
    # for _, base_code in search.get_base_codes():
        selection = (data["group"] == group) & (data["base_code"] == base_code)
        if not any(selection):
            continue
        fig, ax = plot_rates(data[selection], alpha=1)
        ax.set_title(f"{group}, {base_code}")
        fig.tight_layout()
        plt.show()


# Hamming-2:
# - distance <= 6
# - rate ~ [0.10, 0.25]
# Hamming-3:
# - distance ~ large
# - rate <~ 0.15
# CordaroWagner-3:
# - distance <= 6
# - rate ~ [0.15, 0.20]
# CordaroWagner-4:
# - achievable [d, k/n] ~ [10, 0.06]
# CordaroWagner-5:
# - distances ~ 10--20, maybe 30
# - rate ~ [0.05, 0.15]
# CordaroWagner-6:
# - distances ~ 4--10, maybe 15 or 20
# - rate ~ [0.10, 0.23]
# Mittal-4: [not interesting]
# Mittal-5: rate <~ 0.15
# - achievable [d, k/n] ~ [10, 0.08] and maybe [20, 0.05]
# Mittal-6: rate <~ 0.10
# - rates ~ [0.02, 0.10] at d > 10
# - achievable [d, k/n] ~ [25, 0.06]
# - distances ~ 50--100 at k/n >~ 0.15

# (6,2): [d, k/n] ~ [10, 0.05]
# (7,1): [d, k/n] ~ [15, 0.04]
# (8,1): [d, k/n] ~ [20--30, 0.05]
# (8,2): [d, k/n] ~ [12, 0.08]
# (8,3): [d, k/n] ~ [12, 0.10], [15, 0.06]
# (8,5): [d, k/n] ~ [12, 0.15]
# (9,1): k/n ~ 0.05 at d ~ 10--30, maybe 50
# (9,2): [d, k/n] ~ [9, 0.10], k ~ 0.03 at d ~ 30?
# (10,1): [d, k/n] ~ [10, 0.15] or [20, 0.05]; [40--60, 0.05]?
# (10,2): [d, k/n] ~ [13, 0.08], [40--60, 0.03]?
# (11,1): [d, k/n] ~ [20--60, 0.04]
# (11,1): [d, k/n] ~ [20--60, 0.04]
# ...
# (12,5): k/n ~ 0.1 at d ~ 15
# (13,1): k/n ~ 0.04 at d ~ 15--80
# ...
# (16,1) k/n ~ 0.15 at d ~ 10
# (16,4) k/n ~ 0.08 at d ~ 20
# (16,11) k/n ~ 0.1 at d ~ 13
# (16,14) k/n ~ 0.1 at d ~ 15
# ...
# (18,3) k/n ~ 0.05 at d ~ 20--80
# ...
# (19,1) k/n ~ 0.03 at d ~ 20--120
# (20,1) k/n ~ 0.11 at d ~ 20
# ...
# (20,4) k/n ~ 0.11 at d ~ 25
