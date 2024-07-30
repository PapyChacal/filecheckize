#!/usr/bin/env python3

import argparse
import re
import sys
from io import TextIOWrapper
from typing import cast

UNNAMED_SSA_VALUE = re.compile(r"%([\d]+)")
SSA_VALUE_NAME = re.compile(r"%([\d]+|[\w$._-][\w\d$._-]*)(:|#[\d]*)?")
BASIC_BLOCK_NAME = re.compile(r"\^([\d]+|[\w$._-][\w\d$._-]*)")

ANONYMOUS_VARIABLE = r"%{{.*}}"
ANONYMOUS_BLOCK = r"^{{.*}}"

SUBSTITUTED_VARS: "set[str]" = set()
SUBSTITUTED_BLOCKS: "set[str]" = set()


def substitute_variable(var: str) -> str:
    if var not in SUBSTITUTED_VARS:
        SUBSTITUTED_VARS.add(var)
        return f"[[%{var}:%.*]]"
    return f"[[%{var}]]"


def substitute_block(block: str) -> str:
    if block not in SUBSTITUTED_BLOCKS:
        SUBSTITUTED_BLOCKS.add(block)
        return f"[[^{block}:^.*]]"
    return f"[[^{block}]]"


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
        "--substitute",
        action="store_true",
        help="Use variable substituion instead of anonymization.",
    )
    parser.add_argument(
        "--check-empty-lines",
        action="store_true",
        help="Check strictly for empty lines. Default behavior is to just skip them.",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        help="Strip comments from input rather than checking for them.",
    )
    parser.add_argument(
        "--comments-prefix",
        type=str,
        default="//",
        help="Set the prefix of a comment line. Defaults to '//' as in MLIR.",
    )
    parser.add_argument(
        "--check-prefix",
        type=str,
        default="CHECK",
        help="Set the Filecheck prefix to generate. defaults to 'CHECK' as Filecheck's default.",
    )

    args = parser.parse_args(sys.argv[1:])

    comment_line = re.compile(rf"^\s*{re.escape(args.comments_prefix)}.*$")

    if args.strip_comments:

        def strip(line: str) -> bool:
            return bool(re.match(comment_line, line))

    else:

        def strip(line: str) -> bool:
            return False

    if args.mlir_anonymize:

        if args.substitute:

            def anonymize(line: str) -> str:
                # Anonymize SSA value names
                return re.sub(
                    BASIC_BLOCK_NAME,
                    lambda m: substitute_block(m.group(1)),
                    re.sub(
                        SSA_VALUE_NAME, lambda m: substitute_variable(m.group(1)), line
                    ),
                )

        else:

            def anonymize(line: str) -> str:
                # Anonymize SSA value names
                return re.sub(
                    BASIC_BLOCK_NAME,
                    ANONYMOUS_BLOCK,
                    re.sub(SSA_VALUE_NAME, ANONYMOUS_VARIABLE, line),
                )

    elif args.xdsl_anonymize:

        if args.substitute:

            def anonymize(line: str) -> str:
                # Anonymize SSA value names
                return re.sub(
                    BASIC_BLOCK_NAME,
                    lambda m: substitute_block(m.group(1)),
                    re.sub(
                        UNNAMED_SSA_VALUE,
                        lambda m: substitute_variable(m.group(1)),
                        line,
                    ),
                )

        else:

            def anonymize(line: str) -> str:
                # Anonymize SSA value names
                return re.sub(
                    BASIC_BLOCK_NAME,
                    ANONYMOUS_BLOCK,
                    re.sub(UNNAMED_SSA_VALUE, ANONYMOUS_VARIABLE, line),
                )

    else:

        def anonymize(line: str) -> str:
            return line

    prefix = args.check_prefix

    next = False

    for line in cast(TextIOWrapper, args.file):
        line = line.rstrip()

        # Ignore whitespace-only lines
        if not line:
            if args.check_empty_lines:
                print(f"// {prefix}-EMPTY:")
                next = True
            else:
                next = False
            continue

        elif "// -----" in line:
            SUBSTITUTED_VARS.clear()
            SUBSTITUTED_BLOCKS.clear()

        # Ignore remaining comment lines
        if strip(line):
            continue

        line = anonymize(line)

        # Print the modified line
        if next:
            print(f"// {prefix}-NEXT:  ", end="")
        else:
            print(f"// {prefix}:       ", end="")
            next = True
        print(line)
