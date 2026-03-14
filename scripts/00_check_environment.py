import sys
import platform
import importlib

PKGS = ["numpy", "pandas", "matplotlib", "sklearn", "wfdb"]


def get_version(pkg_name: str) -> str:
    try:
        mod = importlib.import_module(pkg_name)
        version = getattr(mod, "__version__", "unknown")
        return version
    except Exception as e:
        error_name = e.__class__.__name__
        return f"NOT INSTALLED ({error_name})"


def main():
    print("=" * 60)
    print("IDSC2026 Team Top Coders - Environment Check")
    print("=" * 60)

    python_version = sys.version.replace("\n", " ")
    platform_name = platform.platform()

    print("Python:", python_version)
    print("Platform:", platform_name)
    print("-" * 60)

    for p in PKGS:
        version = get_version(p)
        print(f"{p:12s}:", version)

    print("=" * 60)
    print("OK: environment check completed.")


if __name__ == "__main__":
    main()
