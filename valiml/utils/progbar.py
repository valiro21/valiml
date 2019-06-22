from progressbar import ProgressBar, UnknownLength
from progressbar import SimpleProgress, DynamicMessage, Bar, AdaptiveETA


def marker(progress, data, width):
    if progress.max_value is not UnknownLength and progress.max_value > 0:
        length = int(progress.value / progress.max_value * width)
        return "".join((['='] * max(0, length - 1) + ['>']))
    else:
        return '='


class ETAFormat:
    def format(self, **data):
        eta_seconds = data['eta_seconds']
        if eta_seconds > 3600:
            eta_format = ('%d:%02d:%02d' %
                          (eta_seconds // 3600, (eta_seconds % 3600) // 60, eta_seconds % 60))
        elif eta_seconds > 60:
            eta_format = '%d:%02d' % (eta_seconds // 60, eta_seconds % 60)
        else:
            eta_format = '%ds' % eta_seconds

        return ' - ETA: %s' % eta_format


class FinishedETAFormat:
    def format(self, **data):
        time_per_unit = data['seconds_elapsed'] / data['value']
        if time_per_unit >= 1:
            return ' %.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
            return ' %.0fms/step' % (time_per_unit * 1e3)
        else:
            return ' %.0fus/step' % (time_per_unit * 1e6)


class ValueFormat:
    def format(self, **data):
        if data['value'] is None:
            return '-'
        return ' - {name} {value:.3f}'.format(**data)


def create_progbar_keras(max_value, stateful_metrics=[]):
    widgets = [
        SimpleProgress(format='{value}/{max_value}', new_style=True), ' ',
        Bar(marker=marker, left='[', right=']', fill='.'),
        AdaptiveETA(
            format_not_started='- ETA: --',
            format_finished=FinishedETAFormat(),
            format=ETAFormat(),
            format_zero=' - ETA: 0s',
            format_NA='- ETA: N/A',
            new_style=True
        )
    ]

    for metric in stateful_metrics:
        widgets.append(DynamicMessage(name=metric, format=ValueFormat()))
    return ProgressBar(max_value=max_value, widgets=widgets, redirect_stdout=True)
