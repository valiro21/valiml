double get_seconds_since_epoch();

void print_char(int c);

int print_float(double x, int max_precision);

int print_bar(int current, int target, double prev_total_width, double width, int dynamic_display);

int print_info_bar(int current, int target, double elapsed_from_start);

void print_remaining(int current, int target, int prev_total_width, int total_width);