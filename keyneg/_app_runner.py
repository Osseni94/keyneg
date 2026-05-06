"""Console-script entry point for the KeyNeg Streamlit app.

Installed as the ``keyneg-app`` command via ``pip install keyneg[app]``.
Resolves ``keyneg/app.py`` via ``importlib.resources`` so it works whether
the package is installed in site-packages, in editable mode, or zipped.
"""

import sys
from importlib import resources


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.stderr.write(
            "Streamlit is not installed. Reinstall with: pip install keyneg[app]\n"
        )
        sys.exit(1)

    # Locate the bundled app module file.
    app_files = resources.files("keyneg")
    app_path = app_files / "app.py"

    # ``resources.files`` may return a Traversable that isn't a real fs path
    # (e.g. inside a zipimport). ``as_file`` materializes it for the duration
    # of the call.
    with resources.as_file(app_path) as concrete_path:
        sys.argv = ["streamlit", "run", str(concrete_path), *sys.argv[1:]]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
