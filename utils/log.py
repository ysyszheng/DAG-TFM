import sys

COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

def log(*args):
    fn = sys._getframe().f_back.f_code.co_filename
    ln = sys._getframe().f_back.f_lineno
    highlight_msg = 'File \"%s\", line %d\n' % (fn, ln)

    colored_msg = COLOR_RED + highlight_msg + COLOR_RESET
    print(colored_msg, *args)

if __name__ == '__main__':
    log('hello world')
