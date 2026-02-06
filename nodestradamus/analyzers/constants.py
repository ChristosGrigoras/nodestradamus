"""Shared constants for analyzers.

Constants that are used across multiple analyzer modules.
"""

import re

# Known standard library modules (top-level)
# Used by impact analysis and graph algorithms to filter external dependencies
STDLIB_MODULES = frozenset({
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio", "asyncore",
    "atexit", "audioop", "base64", "bdb", "binascii", "binhex", "bisect",
    "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
    "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy", "copyreg",
    "cProfile", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
    "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email",
    "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
    "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
    "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "ssl", "stat",
    "statistics", "string", "stringprep", "struct", "subprocess", "sunau",
    "symtable", "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib",
    "tempfile", "termios", "test", "textwrap", "threading", "time", "timeit",
    "tkinter", "token", "tokenize", "trace", "traceback", "tracemalloc", "tty",
    "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
    "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser", "winreg",
    "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile",
    "zipimport", "zlib", "zoneinfo",
    # Also include typing_extensions and common typing-related
    "typing_extensions",
})

# Patterns for identifying test files
# Used to filter test files from various analyses
TEST_FILE_PATTERNS = [
    re.compile(r"test_[^/]+\.py$"),  # test_*.py
    re.compile(r"[^/]+_test\.py$"),  # *_test.py
    re.compile(r"tests?/"),  # tests/ or test/ directory
    re.compile(r"__tests__/"),  # __tests__/ directory
    re.compile(r"\.test\.[tj]sx?$"),  # *.test.ts, *.test.js, etc.
    re.compile(r"\.spec\.[tj]sx?$"),  # *.spec.ts, *.spec.js, etc.
]


def is_test_file(file_path: str) -> bool:
    """Check if a file path is a test file.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file is a test file, False otherwise.
    """
    if not file_path:
        return False
    for pattern in TEST_FILE_PATTERNS:
        if pattern.search(file_path):
            return True
    return False
