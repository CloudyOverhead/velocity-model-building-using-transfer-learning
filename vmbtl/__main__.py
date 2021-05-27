# -*- coding: utf-8 -*-
"""Launch dataset generation, training or testing."""

from GeoFlow.__main__ import main, parser


def parse_args():
    from vmbtl import datasets, architecture

    args, unknown_args = parser.parse_known_args()
    assert not (args.train and args.test)
    args.nn = getattr(architecture, args.nn)
    args.params = getattr(architecture, args.params)
    args.params = args.params(is_training=args.train)
    args.dataset = getattr(datasets, args.dataset)(args.noise)
    for arg, value in zip(unknown_args[::2], unknown_args[1::2]):
        arg = arg.strip('-')
        if arg in args.params.__dict__.keys():
            setattr(args.params, arg, eval(value))
        else:
            raise ValueError(f"Argument `{arg}` not recognized. Could not "
                             f"match it with an existing hyperparameter.")
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
