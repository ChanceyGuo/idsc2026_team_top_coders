import sys
import platform
import importlib

PKGS = ["numpy", "pandas", "matplotlib", "sklearn", "wfdb"]

def get_version(pkg_name: str) -> str:
    try:
        mod = importlib.import_module(pkg_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:
        return f"NOT INSTALLED ({e.__class__.__name__})"

def main():
    print("=" * 60)
    print("IDSC2026 Team Top Coders - Environment Check")
    print("=" * 60)
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("-" * 60)
    for p in PKGS:
        print(f"{p:12s}:", get_version(p))
    print("=" * 60)
    print("OK: environment check completed.")

if __name__ == "__main__":
    main()