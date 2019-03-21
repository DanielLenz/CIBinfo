from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from cibinfo.powerspectra import cibxcib as TT


def make_plot(freq: str, outdir=None) -> None:
    if not isinstance(freq, str):
        raise TypeError('freq must be str')

    model_names = ['Planck14Data', 'Planck14Model', 'Maniyar18Model', 'Mak17']

    _, ax = plt.subplots(figsize=(6, 4))

    for model_name in model_names:
        model = getattr(TT, model_name)(freq, unit='Jy^2/sr')
        try:
            hasattr(model, 'Cl')
        except KeyError:
            continue
            # pass

        if model_name.lower().endswith('model'):
            ax.plot(
                model.l,
                model.Cl,
                label=model_name,
            )
        elif model_name.lower().endswith('data'):
            ax.errorbar(
                model.l,
                model.Cl,
                yerr=model.dCl,
                marker='.',
                capsize=6,
                linestyle='None',
                label=model_name,
            )

        # Log-log
        ax.loglog()

        # Limits
        ax.set_xlim(40, 2000)

        # Labels
        ax.legend()
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$C_{\ell}$')

        if outdir is not None:
            outdir.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                outdir.joinpath(f'autopower_comparison_{freq}GHz.pdf'),
                dpi=300)


def main():
    outdir = Path('figures/')
    freqs = set(['353', '545', '857'])

    for freq in freqs:
        make_plot(freq, outdir=outdir)


if __name__ == "__main__":
    main()