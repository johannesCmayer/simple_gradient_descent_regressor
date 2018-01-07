import sys
import time


class ProgressIndicator:
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    def __init__(self, total, prefix='', suffix='', decimals=1, bar_length=100):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.bar_length = bar_length

        self.start_time = 0
        self.previous_time = time.time()
        self.execution_times = []

    def advance_iter(self, iteration, total):
        if iteration is 0:
            self.start_time = time.time()

        str_format = "{0:." + str(self.decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(self.bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (self.bar_length - filled_length)

        estimated_completion_time = 0
        if len(self.execution_times) != 0:
            estimated_completion_time = sum(self.execution_times) / len(self.execution_times) * (total - len(self.execution_times))

        caller_name = sys._getframe(1).f_code.co_name
        sys.stdout.write('\r{} x {} |{}|{}{} ETC-{} {}'.format(self.prefix, caller_name, bar, percents, '%', truncate(estimated_completion_time, 1), self.suffix)),

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

        self.execution_times.append(time.time() - self.previous_time)
        self.previous_time = time.time()

    def print_total_execution_time(self):
        print('\nTotal Execution Time: {}'.format(self.total_execution_time()))

    def total_execution_time(self):
        return sum(self.execution_times)


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])
