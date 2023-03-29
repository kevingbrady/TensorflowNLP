import sys


class ArgumentManager:

    args = []

    def __init__(self):
        self.args = sys.argv

        self.sp800_53 = next(x for x in self.args if x.__contains__('800-53'))
        self.catalog = next(x for x in self.args if x.__contains__('800-213A'))
        self.csf = next(x for x in self.args if x.__contains__('cybersecurityframework'))
