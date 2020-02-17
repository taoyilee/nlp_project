#  MIT License
#  #
#  Copyright (c) 2020, Michael Tao-Yi Lee
#  #
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  #
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from configobj import ConfigObj


class DefaultSection:
    def __init__(self, config_obj):
        self.config_obj = config_obj

    @property
    def output_dir(self) -> str:
        return self.config_obj["output_dir"]


class WikiTextSection:
    def __init__(self, config_obj):
        self.config_obj = config_obj

    @property
    def train_file(self) -> str:
        return self.config_obj["train_file"]

    @property
    def test_file(self) -> str:
        return self.config_obj["test_file"]


class Configuration:
    def __init__(self, infile=None):
        import os
        if infile is None:
            infile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini")
            if not os.path.isfile(infile):
                raise FileNotFoundError(f"{infile} does not exist")
        self.config_obj = ConfigObj(infile)

    def write(self, outfile=None):
        if outfile is not None:
            self.config_obj.filename = outfile
        self.config_obj.write()

    @property
    def default(self):
        return DefaultSection(self.config_obj["DEFAULT"])

    @property
    def wikitext(self):
        return WikiTextSection(self.config_obj["WikiText-2"])
