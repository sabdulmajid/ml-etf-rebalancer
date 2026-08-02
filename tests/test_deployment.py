import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_DIRECTORY = ROOT / "dashboard"
DEPENDENCY_FILENAMES = (
    "uv.lock",
    "Pipfile",
    "environment.yml",
    "requirements.txt",
    "pyproject.toml",
)
REQUIRED_RUNTIME_DISTRIBUTIONS = {
    "exchange-calendars",
    "numpy",
    "pandas",
    "plotly",
    "scikit-learn",
    "streamlit",
    "yfinance",
}


def _declared_distributions(requirements_path):
    declared = set()
    for line in requirements_path.read_text().splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith(("-", "http://", "https://")):
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match:
            declared.add(match.group(0).lower().replace("_", "-"))
    return declared


def test_streamlit_cloud_uses_the_root_dependency_manifest_only():
    """Prevent an entrypoint-local file from shadowing root requirements.

    Streamlit Community Cloud searches beside ``dashboard/app.py`` before the
    repository root. A second manifest here previously omitted
    ``exchange-calendars`` and broke production at import time.
    """
    root_requirements = ROOT / "requirements.txt"
    assert (ENTRYPOINT_DIRECTORY / "app.py").is_file()
    dependency_files = [
        path
        for directory in (ENTRYPOINT_DIRECTORY, ROOT)
        for name in DEPENDENCY_FILENAMES
        if (path := directory / name).exists()
    ]
    assert dependency_files == [root_requirements]


def test_root_manifest_declares_dashboard_runtime_dependencies():
    declared = _declared_distributions(ROOT / "requirements.txt")
    assert REQUIRED_RUNTIME_DISTRIBUTIONS <= declared
