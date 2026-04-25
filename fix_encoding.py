"""
Fix Windows console encoding to support UTF-8 and emoji.

Import this module FIRST in every Python script to prevent
UnicodeEncodeError crashes caused by emoji in print statements
on Windows terminals with cp1252 encoding.

Usage:
    import fix_encoding  # Must be first import
"""

import sys
import io
import os


def setup_utf8_stdout():
    """Reconfigure stdout/stderr to UTF-8 with error replacement."""
    if os.name != 'nt':
        return  # Only needed on Windows

    try:
        # Python 3.7+ recommended approach
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        else:
            # Fallback for older Python
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace'
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding='utf-8', errors='replace'
            )
    except Exception:
        pass  # If all else fails, continue silently


# Run immediately on import
setup_utf8_stdout()
