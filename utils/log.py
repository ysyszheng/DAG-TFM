import sys

def log(msg):
    fn = sys._getframe().f_back.f_code.co_filename
    ln = sys._getframe().f_back.f_lineno
    print('File \"%s\", line %d\n' % (fn, ln), msg)
