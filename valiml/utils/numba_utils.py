import importlib

import numba
import numpy as np
from numba import types
from numba.typed import Dict
from numba.cffi_support import register_module

loaded = importlib.import_module('_utils')
register_module(loaded)

get_seconds_since_epoch = loaded.lib.get_seconds_since_epoch
print_char_int = loaded.lib.print_char
print_float = loaded.lib.print_float

print_bar = loaded.lib.print_bar
print_info_bar = loaded.lib.print_info_bar
print_remaining = loaded.lib.print_remaining


@numba.jit(nopython=True)
def numba_ord(chr):
    if chr == '=':
        return 61
    elif chr == '.':
        return 46
    elif chr == '>':
        return 62
    elif chr == '[':
        return 91
    elif chr == ']':
        return 93
    elif chr == '\r':
        return 13
    elif chr == '\b':
        return 8
    elif chr == '\n':
        return 10
    elif chr == '_':
        return 95
    elif chr == ' ':
        return 32
    elif chr == '/':
        return 47
    elif chr == '+':
        return 43
    elif chr == '-':
        return 45
    elif chr == '*':
        return 42
    elif chr == ':':
        return 58
    elif chr == '{':
        return 123
    elif chr == '}':
        return 125
    elif chr == '0':
        return 48
    elif chr == '1':
        return 49
    elif chr == '2':
        return 50
    elif chr == '3':
        return 51
    elif chr == '4':
        return 52
    elif chr == '5':
        return 53
    elif chr == '6':
        return 54
    elif chr == '7':
        return 55
    elif chr == '8':
        return 56
    elif chr == '9':
        return 57
    elif chr == 'a':
        return 97
    elif chr == 'b':
        return 98
    elif chr == 'c':
        return 99
    elif chr == 'd':
        return 100
    elif chr == 'e':
        return 101
    elif chr == 'f':
        return 102
    elif chr == 'g':
        return 103
    elif chr == 'h':
        return 104
    elif chr == 'i':
        return 105
    elif chr == 'j':
        return 106
    elif chr == 'k':
        return 107
    elif chr == 'l':
        return 108
    elif chr == 'm':
        return 109
    elif chr == 'n':
        return 110
    elif chr == 'o':
        return 111
    elif chr == 'p':
        return 112
    elif chr == 'q':
        return 113
    elif chr == 'r':
        return 114
    elif chr == 's':
        return 115
    elif chr == 't':
        return 116
    elif chr == 'u':
        return 117
    elif chr == 'v':
        return 118
    elif chr == 'w':
        return 119
    elif chr == 'x':
        return 120
    elif chr == 'y':
        return 121
    elif chr == 'z':
        return 122
    elif chr == 'A':
        return 65
    elif chr == 'B':
        return 66
    elif chr == 'C':
        return 67
    elif chr == 'D':
        return 68
    elif chr == 'E':
        return 69
    elif chr == 'F':
        return 70
    elif chr == 'G':
        return 71
    elif chr == 'H':
        return 72
    elif chr == 'I':
        return 73
    elif chr == 'J':
        return 74
    elif chr == 'K':
        return 75
    elif chr == 'L':
        return 76
    elif chr == 'M':
        return 77
    elif chr == 'N':
        return 78
    elif chr == 'O':
        return 79
    elif chr == 'P':
        return 80
    elif chr == 'Q':
        return 81
    elif chr == 'R':
        return 82
    elif chr == 'S':
        return 83
    elif chr == 'T':
        return 84
    elif chr == 'U':
        return 85
    elif chr == 'V':
        return 86
    elif chr == 'W':
        return 87
    elif chr == 'X':
        return 88
    elif chr == 'Y':
        return 89
    elif chr == 'Z':
        return 90
    return -1


@numba.jit(nopython=True)
def print_string(s):
    for idx in range(len(s)):
        character = s[idx]
        print_char_int(numba_ord(character))

    return len(s)


@numba.jitclass([
    ('target', types.int32),
    ('width', types.int32),
    ('verbose', types.int8),
    ('interval', types.float32),
    ('_dynamic_display', types.boolean),
    ('_metrics_sum', Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )._numba_type_),
    ('_metrics_count', Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int32,
    )._numba_type_),
    ('_total_width', types.int32),
    ('_seen_so_far', types.int32),
    ('_start', types.float64),
    ('_last_update', types.float64)
])
class ProgressBar:
    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None, dynamic_display=True):
        """
        Displays a pretty progress bar on screen. Inspired from keras.utils.Progbar

        :param target: The total number of steps expected, None if unknown.
        :param width: Progress bar width on screen.
        :param verbose: Verbosity: 0 (silent), 1 (verbose)
        :param interval: Minimum visual progress update interval (in seconds).
        :param stateful_metrics: Iterable of names of the metrics that should not be averaged over time.

        `>>> bar = ProgresBar(10, stateful_metrics=['error', 'cpu_usage'])`
        """
        self.target = -1 if target is None else target
        self.width = width
        self.verbose = verbose
        self.interval = interval

        self._metrics_sum = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64
        )
        self._metrics_count = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.int32
        )

        for metric_name in stateful_metrics.split('|'):
            if metric_name != '':
                self._metrics_sum[metric_name] = 0
                self._metrics_count[metric_name] = -1

        self._dynamic_display = dynamic_display
        self._total_width = 0
        self._seen_so_far = 0
        self._start = get_seconds_since_epoch()
        self._last_update = 0

    def update_metric(self, current, metric_name, metric_value):
        if metric_name not in self._metrics_count:
            self._metrics_sum[metric_name] = metric_value
            self._metrics_count[metric_name] = current - self._seen_so_far
        else:
            metric_count = self._metrics_count[metric_name]
            if metric_count == -1:
                self._metrics_sum[metric_name] = metric_value
            else:
                metric_sum = self._metrics_sum[metric_name]
                self._metrics_sum[metric_name] = metric_sum + metric_value
                self._metrics_count[metric_name] = metric_count + current - self._seen_so_far

    def update_metrics(self, current, metrics="", values=None):
        if values is None:
            return

        for metric, value in zip(metrics.split("|"), values):
            self.update_metric(current, metric, value)

    def print_metric(self, metric_name):
        characters_printed = 0

        characters_printed += print_string(' - ')
        characters_printed += print_string(metric_name)
        characters_printed += print_string(':')

        metric_sum = self._metrics_sum[metric_name]
        metric_count = self._metrics_count[metric_name]

        if metric_count > -1:
            avg = metric_sum / max(1., metric_count)
            if np.abs(avg) > 1e-3:
                characters_printed += print_float(avg, 4)
            else:
                characters_printed += print_float(avg, 4)
        else:
            characters_printed += print_string(' ')
            characters_printed += print_float(metric_sum, 8)
        return characters_printed

    def print_metrics(self):
        characters_printed = 0
        for metric_name in self._metrics_sum:
            characters_printed += self.print_metric(metric_name)
        return characters_printed

    def update(self, current: int, metrics="", values=None):
        """
        Update the current step and values.

        :param current: The current step index.
        :param values: The values to update.
        :return:
        """
        self.update_metrics(current, metrics, values)
        self._seen_so_far = current

        now = get_seconds_since_epoch()
        elapsed_from_start = now - self._start
        elapsed_from_last_update = now - self._last_update

        if self.verbose == 1:
            if elapsed_from_last_update < self.interval and self.target != -1 and current < self.target:
                return

            prev_total_width = self._total_width
            self._total_width = print_bar(current, self.target, prev_total_width, self.width, self._dynamic_display)

            info_len = print_info_bar(current, self.target, elapsed_from_start)
            info_len += self.print_metrics()

            self._total_width += info_len

            print_remaining(current, self.target, prev_total_width, self._total_width)

        self._last_update = now

    def add(self, n, metrics="", values=None):
        self.update(self._seen_so_far + n, metrics, values)


@numba.jit(nopython=True, cache=True)
def create_progbar_numba(target, width=30, verbose=1, interval=0.05, stateful_metrics="", dynamic_display=True):
    return ProgressBar(
        target,
        width,
        verbose,
        interval,
        stateful_metrics,
        dynamic_display
    )
