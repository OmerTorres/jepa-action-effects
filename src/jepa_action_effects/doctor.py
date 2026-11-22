"""Check the numerical environment without fetching models or simulator data."""

import importlib
import io
import json
import platform
import sys
from importlib.metadata import version


def diagnose() -> dict:
    """Return measured runtime facts; fail if a required import or calculation fails."""
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "pyyaml": "yaml",
    }
    for module in packages.values():
        importlib.import_module(module)

    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.linear_model import Ridge

    # Exercise native numerical libraries and headless rendering, not just metadata.
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * x[:, 0] + 1
    model = Ridge(alpha=0.0).fit(x, y)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-10)
    figure = Figure(figsize=(2, 2))
    FigureCanvasAgg(figure)
    figure.subplots().plot(x[:, 0], y)
    png = io.BytesIO()
    figure.savefig(png, format="png")
    if not png.getvalue().startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Headless rendering did not produce a PNG")

    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {package: version(package) for package in packages},
        "checks": {"ridge_fit": "passed", "headless_png": "passed"},
        "study_readiness": (
            "PHYS-002, SHIFT-002 and H3-PLAN-002 are complete; see STUDY_SYNTHESIS.md "
            "and the REVIEW-BRANCH-002 corrections (D026). New sealed runners must use "
            "provenance validation before confirm."
        ),
    }


def main() -> None:
    """Emit an environment report suitable for an experiment manifest."""
    print(json.dumps(diagnose(), indent=2))


if __name__ == "__main__":
    main()
