# _*_coding:utf-8_*_

import pickle
import argparse
import pprint
import kenlm
import numpy

from converter.RedirectConverter import RedirectConverter

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
        required=True,
        type=float,
    )

    parser.add_argument(
        "-t", "--target",
        required=True,
        type=argparse.FileType("r"),
        dest="target_file"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=argparse.FileType("w"),
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    model = kenlm.Model(VECTOR_FILE)

    lexer = Simplexer()

    tokenbook = lexer.create_tokenbook(args.target_file)
    print len(tokenbook)
    codebook = lexer.create_codebook(tokenbook)
    print len(codebook)

    bag_of_ngrams = ngrams(codebook, args.N)

    line = 0
    detect_count = 0
    x = []
    y = []
    detect_line = []
    for ngram in bag_of_ngrams:
        perplexity = 10**(-model.score(" ".join(ngram)) / args.N + 1)
        probabilty = 1/perplexity
        print "{0}: {1} ---> {2:>20}".format(line, ngram, probabilty)
        x.append(line)
        y.append(probabilty)
        if args.threshould < probabilty:
            detect_line.append(tokenbook[line].lineno)

        line += 1

    x = numpy.array(x)
    y = numpy.array(y)
    z = numpy.polyfit(x, y, args.dimention)

    p = numpy.poly1d(z)
    xp = numpy.linspace(0, 100, 100)

    # plt.plot(x, y, '.', xp, p(xp))
    plt.plot(x, y, '.')

    plt.ylim(0, 0.1)
    plt.show()

    count = 1
    converted_code = []
    rc = RedirectConverter()
    args.target_file.seek(0)
    print set(detect_line)
    for l in args.target_file:
        if count in set(detect_line):
            print "Convert: {}".format(l)
            converted_code.append(rc.convert(l))
        else:
            converted_code.append(l)
        count += 1

    if args.output_file is None:
        for l in converted_code:
            print l
    else:
        args.output_file.writelines(converted_code)
