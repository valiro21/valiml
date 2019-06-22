import sys

from valiml.utils.progbar import create_progbar_keras


def normalize(x):
    return x / x.sum()


def create_progbar(target, stateful_metrics=None, numba=False):
    if numba:
        from valiml.utils.numba_utils import ProgressBar

        dynamic_display = ((hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()) or 'ipykernel' in sys.modules)
        stateful_metrics = '|'.join(set(stateful_metrics)) if stateful_metrics else ''

        return ProgressBar(target,
                           width=30,
                           verbose=1,
                           stateful_metrics=stateful_metrics,
                           dynamic_display=dynamic_display
                           )
    else:
        return create_progbar_keras(target, stateful_metrics=stateful_metrics)
