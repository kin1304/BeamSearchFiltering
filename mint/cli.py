#!/usr/bin/env python3
"""
Main CLI for Mint Module - Run graph and beam search commands
"""

import argparse
import sys
import os

def run_graph_command(args):
    """Run graph CLI command"""
    from mint.graph.cli import main as graph_main
    
    # Set sys.argv for graph CLI
    sys.argv = ['graph_cli'] + args
    try:
        graph_main()
    except SystemExit:
        pass

def run_beam_command(args):
    """Run beam search CLI command"""
    from mint.beam_search.cli import main as beam_main
    
    # Set sys.argv for beam CLI
    sys.argv = ['beam_cli'] + args
    try:
        beam_main()
    except SystemExit:
        pass

def run_filtering_command(args):
    """Run filtering CLI command"""
    from .filtering.cli import main as filtering_main
    
    # Set sys.argv for filtering CLI
    sys.argv = ['filtering_cli'] + args
    try:
        filtering_main()
    except SystemExit:
        pass

def main():
    parser = argparse.ArgumentParser(description='Mint Module CLI')
    parser.add_argument('module', choices=['graph', 'beam', 'filtering'], help='Module to run')
    parser.add_argument('command', help='Command to run')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Additional arguments')
    
    args = parser.parse_args()
    
    if args.module == 'graph':
        run_graph_command([args.command] + args.args)
    elif args.module == 'beam':
        run_beam_command([args.command] + args.args)
    elif args.module == 'filtering':
        run_filtering_command([args.command] + args.args)

if __name__ == '__main__':
    main() 