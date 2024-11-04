#!/usr/bin/env python3

import argparse
import re
import sys
from io import TextIOWrapper
from typing import cast


def main():
    parser = argparse.ArgumentParser(
        description="Generate the FileCheck lines that expects the given input."
    )
    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        default=sys.stdin,
        nargs="?",
        help="Input file to read. Defaults to standard input.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--compact-output",
        action="store_true",
        help="Do not print empty lines between streaks of CHECK-NEXT.",
    )
    group.add_argument(
        "--mlir-anonymize",
        action="store_true",
        help="Anonymize MLIR SSA value names and basic block names.",
    )
    group.add_argument(
        "--xdsl-anonymize",
        action="store_true",
        help="Anonymize MLIR unnamed SSA values and basic block names.",
    )
    parser.add_argument(
        "--check-empty-lines",
        action="store_true",
        help="Check strictly for empty lines. Default behavior is to just skip them.",
    )
    parser.add_argument(
        "--strip-comments",
        type=str,
        nargs="?",
        default="",
        const="//",
        help="Strip comments from input rather than checking for them. Can take a custom comment prefix, otherwise defaults to // as in MLIR",
    )
    parser.add_argument(
        "--comments-prefix",
        type=str,
        default="//",
        help="Set the comment prefix to generate. defaults to '//' as in MLIR.",
    )
    parser.add_argument(
        "--check-prefix",
        type=str,
        default="CHECK",
        help="Set the Filecheck prefix to generate. defaults to 'CHECK' as Filecheck's default.",
    )

    args = parser.parse_args(sys.argv[1:])

    comment_line = re.compile(rf"^\s*{re.escape(args.strip_comments)}.*$")
    unnamed_ssa_value = re.compile(r"%([\d]+)")
    ssa_value_name = re.compile(r"%([\d]+|[\w$._-][\w\d$._-]*)(:|#[\d]*)?")
    basic_block_name = re.compile(r"\^([\d]+|[\w$._-][\w\d$._-]*)")

    prefix = args.check_prefix
    comm = args.comments_prefix

    next = False

    for line in cast(TextIOWrapper, args.file):
        line = line.rstrip()

        # Ignore whitespace-only lines
        if not line:
            if args.check_empty_lines:
                print(f"{comm} {prefix}-EMPTY:")
                next = True
            else:
                # Print empty lines between streaks of CHECK-NEXT
                if next and not args.compact_output:
                    print("")
                next = False
            continue

        # Ignore remaining comment lines
        if args.strip_comments:
            if re.match(comment_line, line):
                continue

        if args.mlir_anonymize or args.xdsl_anonymize:
            if args.mlir_anonymize:
                # Anonymize SSA value names
                line = re.sub(ssa_value_name, r"%{{.*}}", line)
            elif args.xdsl_anonymize:
                # Anonymize unnamed SSA values
                line = re.sub(unnamed_ssa_value, r"%{{.*}}", line)

            # Anonymize basic blocks names
            line = re.sub(basic_block_name, r"^{{.*}}", line)

        # Print the modified line
        if next:
            print(f"{comm} {prefix}-NEXT:  ", end="")
        else:
            print(f"{comm} {prefix}:       ", end="")
            next = True
        print(line)
