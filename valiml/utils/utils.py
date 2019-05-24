import sys


def normalize(x):
    return x / x.sum()


def create_progbar(target, width=30, verbose=1, interval=0.05, stateful_metrics=None, numba=True):
    if numba:
        from valiml.utils.numba_utils import ProgressBar

        dynamic_display = ((hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()) or 'ipykernel' in sys.modules)
        stateful_metrics = '|'.join(set(stateful_metrics)) if stateful_metrics else ''

        return ProgressBar(target,
                           width=width,
                           verbose=verbose,
                           interval=interval,
                           stateful_metrics=stateful_metrics,
                           dynamic_display=dynamic_display
                           )
