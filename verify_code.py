# _*_coding:utf-8_*_

import pickle
import argparse
import pprint
import kenlm
import numpy

import matplotlib.pyplot as plt

from Simplexer import Simplexer

from nltk.util import ngrams

VECTOR_FILE = "database/data_for_kenlm.dat.arpa"


def getArgs():
    """
    コマンド引数をパースします
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-n",
        dest="N",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-d",
        dest="dimention",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--threshould",
        dest="threshould",
        # required=True,
        type=float,
    )

    parser.add_argument(
        "-t", "--target",
        required=True,
        type=argparse.FileType("r"),
        dest="target_file"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = getArgs()

    model = kenlm.Model(VECTOR_FILE)
    print dir(model)

    lexer = Simplexer()

    letterbook = lexer.analyze(args.target_file)

    bag_of_ngrams = ngrams(letterbook, args.N)

    line = 0
    detect_count = 0
    x = []
    y = []
    for ngram in bag_of_ngrams:
        perplexity = 10**(-model.score(" ".join(ngram)) / args.N + 1)
        probabilty = 1/perplexity
        print "{0}: {1} ---> {2:>20}".format(line, ngram, probabilty)
        x.append(line)
        y.append(probabilty)
        line += 1
    x = numpy.array(x)
    y = numpy.array(y)
    z = numpy.polyfit(x, y, args.dimention)

    p = numpy.poly1d(z)
    xp = numpy.linspace(0, 80, 100)

    plt.plot(x, y, '.', xp, p(xp))

    plt.ylim(0, 0.1)
    plt.show()
