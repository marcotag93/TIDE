"""Allow TIDE to be invoked as ``python -m tide``.

This provides an interpreter-explicit fallback when multiple Python
environments expose different ``tide`` console scripts on ``PATH``.
"""

from tide.cli import main

if __name__ == "__main__":
    main()
