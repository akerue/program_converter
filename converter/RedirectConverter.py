# _*_coding:utf-8_*_

import re
from Converter import Converter


class RedirectConverter(Converter):
    def __init__(self):
        self.pattern = re.compile(r"(?P<input>\S*)\s>\s(?P<output>\S*)")

    def convert(self, line):
        code = self.pattern.search(line)
        return line.replace(" {} > {}".format(code.group("input"),
                                             code.group("output")),
                            "({}, {})".format(code.group("input"),
                                              code.group("output")))
