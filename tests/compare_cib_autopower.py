import numpy as np
import matplotlib.pyplot as plt

from cibinfo.powerspectra import cibxcib as TT


def make_plot(freq):
    if not isinstance(freq, str):
        raise TypeError('freq must be str')

    model_names = ['Maniyar18Model', 'Planck14Data', 'Planck14Model']

    _, ax = plt.subplots(figsize=(6, 4))

    for model_name in model_names:
        model = getattr(TT, model_name)(freq, unit='Jy^2/sr')

        if model_name.lower().endswith('model'):
            ax.plot(
                model.l,
                model.Cl,
                label=model_name,
            )
        elif model_name.lower().endswith('data'):
            ax.scatter(
                model.l,
                model.Cl,
                marker='x',
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

def main():
    freq = '353'
    make_plot(freq)

    plt.savefig(f'autopower_comparison_{freq}GHz.pdf', dpi=300)


if __name__ == "__main__":
    main()