from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from cibinfo.powerspectra import cibxcib as TT

UNIT_STR = {
    "MJy^2/sr": r"$\ [\rm MJy^2/sr$]",
    "Jy^2/sr": r"$\ [\rm Jy^2/sr$]",
    "K^2.sr": r"$\ [\rm K^2\,sr$]",
    "uK^2.sr": r"$\ [\rm \mu K^2\,sr$]",
}


def make_plot(
    freq: str,
    unit: str = "Jy^2/sr",
    ell_scaling="l0",
    logx=False,
    logy=False,
    outdir=None,
) -> None:
    if not isinstance(freq, str):
        raise TypeError("freq must be str")

    model_names = ["Planck14Model", "Maniyar18Model", "Mak17Model", "Planck14Data"]
    # model_names = ["Planck14Model", "Mak17Model", "Planck14Data"]

    _, ax = plt.subplots(figsize=(6, 4))

    for model_name in model_names:
        if model_name == "Mak17Model":
            model = getattr(TT, model_name)(freq, mask="mask40", unit=unit)
        else:
            model = getattr(TT, model_name)(freq, unit=unit)
        try:
            hasattr(model, "Cl")
        except KeyError:
            continue
            # pass

        # Scale the effective Cl. We use either Cl, l Cl, or l^2 Cl / 2pi
        if ell_scaling == "l0":
            cl_eff = model.Cl
            if model.dCl is not None:
                dcl_eff = model.dCl
        elif ell_scaling == "l1":
            cl_eff = model.l * model.Cl
            if model.dCl is not None:
                dcl_eff = model.l * model.dCl
        elif ell_scaling == "l2":
            cl_eff = model.l ** 2 * model.Cl / 2.0 / np.pi
            if model.dCl is not None:
                dcl_eff = model.l ** 2 / 2.0 / np.pi * model.dCl

        # Use a line plot for models
        if model_name.lower().endswith("model"):

            ax.plot(model.l, cl_eff, label=model_name)
        # Use a scatter/errorbar plot for data
        elif model_name.lower().endswith("data"):
            ax.errorbar(
                model.l,
                cl_eff,
                yerr=dcl_eff,
                marker=".",
                capsize=6,
                linestyle="None",
                label=model_name,
            )

        # Possibly use logscales
        if logx:
            ax.semilogx()
        if logy:
            ax.semilogy()

        # Limits
        ax.set_xlim(10, 2100)
        ax.set_ylim(0.5 * np.min(cl_eff), 1.5 * np.max(cl_eff))

        # Grid
        ax.grid()

        # Labels
        ax.legend()
        ax.set_xlabel(r"$\ell$")
        ylabels = {
            "l0": r"$C_{\ell}$" + UNIT_STR[unit],
            "l1": r"$\ell\;C_{\ell}$" + UNIT_STR[unit],
            "l2": r"$\ell^2\,/2\pi\;C_{\ell}$" + UNIT_STR[unit],
        }
        ax.set_ylabel(ylabels[ell_scaling])

        logstr_dict = {
            (False, False): "",
            (True, False): "_logx",
            (True, True): "_loglog",
            (False, True): "_logy",
        }

        logstr = logstr_dict[(logx, logy)]

        if outdir is not None:
            outdir.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                outdir.joinpath(
                    f"autopower_comparison_{freq}GHz_{ell_scaling}{logstr}.pdf"
                ),
                dpi=300,
            )


def main():
    outdir = Path("figures/")
    unit = "uK^2.sr"
    freqs = set(["353", "545", "857"])

    for freq in freqs:
        make_plot(
            freq, unit=unit, ell_scaling="l2", logx=False, logy=False, outdir=outdir
        )


if __name__ == "__main__":
    main()
