"""Generally useful utility functions."""
from __future__ import print_function

import sys

import tensorflow as tf


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()

def test_global_variables():
    params = tf.global_variables()
    print_out("# global variables")
    print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
        print_out("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))

def test_trainable_variables():
    params = tf.trainable_variables()
    print_out("# trainable variables")
    print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
        print_out("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))