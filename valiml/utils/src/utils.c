#include <stdio.h>
#include <math.h>

#if defined(_WIN64) || defined(_WIN32)
#include <Windows.h>
#include <Winsock.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#else
#include <sys/time.h>
#endif

double get_seconds_since_epoch() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

void print_char(int c) {
    char ch = (char)c;
    printf("%c", ch);
}

int print_float(double x, int max_precision) {
    if (max_precision == -1) {
        return printf("%f", x);
    }

    char snum[21];
    sprintf(snum, "%%.%df", max_precision);
    return printf(snum, x);
}

int print_char_repeat(char c, int nr) {
    for (int idx = 0; idx < nr; idx ++) {
        printf("%c", c);
    }
    return nr;
}

int print_bar(int current, int target, int prev_total_width, int width, int dynamic_display) {
    if (dynamic_display) {
        print_char_repeat('\b', prev_total_width);
        printf("\r");
    }
    else {
        printf("\n");
    }

    int bar_len = 0;

    if (target != -1) {
        int numdigits = floor(log10(target)) + 1;

        bar_len += printf("%*d/%d [", numdigits, current, target);

        double prog = (double)current / (double)target;
        int prog_width = (int)((double)width * prog);

        if (prog_width > 0) {
            bar_len+= print_char_repeat('=', prog_width - 1);

            if (current < target) {
                bar_len += printf(">");
            }
            else {
                bar_len += printf("=");
            }
        }

        bar_len += print_char_repeat('.', width - prog_width);
        bar_len += printf("]");
    }
    else {
        bar_len += printf("%*d/Unknown", 7, current);
    }

    return bar_len;
}

int print_info_bar(int current, int target, double elapsed_from_start) {
    int info_len = printf(" - %.0fs", elapsed_from_start);

    double time_per_unit = 0;
    if (current > 0) {
        time_per_unit = elapsed_from_start / (double)current;
    }

    if (target != -1 && current < target) {
        double eta = time_per_unit * (double)(target - current);
        info_len += printf(" - ETA: ");

        if (eta > 3600) {
            int hours = eta / 3600;
            int minutes = ((int)eta % 3600) / 60;
            int seconds = (int)eta % 60;
            info_len += printf("%d:%02d:%02d", hours, minutes, seconds);
        }
        else if (eta > 60) {
            int minutes = eta / 60;
            int seconds = (int)eta % 60;
            info_len += printf("%d:%02d", minutes, seconds);
        }
        else {
            int seconds = eta;
            info_len += printf("%ds", seconds);
        }
    }
    else {
        info_len += printf(" ");

        if (time_per_unit >= 1) {
            info_len += printf("%.0fs/step", time_per_unit);
        }
        else if (time_per_unit >= 1e-3) {
            info_len += printf("%.0fms/step", time_per_unit * 1e3);
        }
        else {
            info_len += printf("%.0fus/step", time_per_unit * 1e6);
        }
    }

    return info_len;
}

void print_remaining(int current, int target, int prev_total_width, int total_width) {
    if (prev_total_width > total_width) {
        print_char_repeat(' ', prev_total_width - total_width);

        if(target != -1 && current >= target) {
            printf("\n\n");
        }
    }

    fflush(stdout);
}
