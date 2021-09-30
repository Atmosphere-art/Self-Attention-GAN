import sys

import utils

from parameters import *
from sagan_models import Generator, Discriminator
from tester import Tester


if __name__ == '__main__':
    config = get_parameters()
    config.command = 'python ' + ' '.join(sys.argv)
    print(config)

    tester = Tester(config)
    tester.test()

