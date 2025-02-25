import sys
import argparse
from utils.conf import Configuration
from utils.expe import Experiment


def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters for LLM4SSS')
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    args = parser.parse_args(argv[1: ])
    conf = Configuration(args.confpath)
    expe = Experiment(conf)
    expe.run()

if __name__ == '__main__':
    main(sys.argv)