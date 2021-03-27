import os
import sys
import time


def print_time(final_time, start_time, msg=None):
    print("--- %s seconds ---"%(final_time - start_time),end='')
    if msg is not None:
        print(msg)
    else:
        print()