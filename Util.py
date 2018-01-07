import sys
import time

class ProgressIndicator:

    def __init__(self, iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        self.iteration = iteration
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.bar_length = bar_length

    def advance_iter(self):
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
        str_format = "{0:." + str(self.decimals) + "f}"
        percents = str_format.format(100 * (self.iteration / float(self.total)))
        filled_length = int(round(self.bar_length * self.iteration / float(self.total)))
        bar = '█' * filled_length + '-' * (self.bar_length - filled_length)

        sys.stdout.write('\r{0} |{1}| {2}{3} ECT-{4} {5}'.format(self.prefix, bar, percents, '%', etc, self.suffix))

        if self.iteration == 0:
            self.start_time = time.time()

        if self.iteration == self.total:
            sys.stdout.write('\n Execution Time: {}'.format(time.time() - self.start_time))
            print()
        sys.stdout.flush()


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])
