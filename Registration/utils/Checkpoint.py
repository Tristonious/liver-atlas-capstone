# Tristan Jones 
# Spring 2026 Capstone 
#
# AI Use Disclosure — LEGACY FILE (not used in current pipeline)
#   Student estimate: 30% student-designed, 70% AI-assisted implementation
#   Claude assisted with: full pickle-based save/load/exists/clear implementation
#   See: "Documentation/AI Use Disclosure.md" for full details


"""
utils/checkpoint.py — Save and load intermediate pipeline stage results.

Each stage output is stored as a .pkl file in the output directory.
This lets you re-run the pipeline from any stage without redoing earlier
expensive steps (e.g. landmark extraction or TPS fitting).
"""

import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)


class Checkpoint:
    """
    Simple key-value store that persists Python objects to disk via pickle.

    Usage:
        cp = Checkpoint(Path("outputs/0010_to_0004"))
        cp.save("landmarks_ref", ref_landmarks_array)
        lm = cp.load("landmarks_ref")      # numpy array back
        cp.exists("landmarks_ref")         # True
    """

    def __init__(self, directory: Path):
        """Helper for init."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        """Helper for path."""
        return self.directory / f"checkpoint_{name}.pkl"

    def exists(self, name: str) -> bool:
        """Return True if a checkpoint file exists for this stage name."""
        return self._path(name).exists()

    def save(self, name: str, value) -> None:
        """
        Serialize value to disk.

        Args:
            name:  Stage identifier string (e.g. "landmarks_ref")
            value: Any pickle-able Python object (numpy array, dict, etc.)
        """
        path = self._path(name)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
            log.debug(f"  Checkpoint saved: {path.name}")
        except Exception as e:
            log.warning(f"  Could not save checkpoint '{name}': {e}")

    def load(self, name: str):
        """
        Load a previously saved checkpoint.

        Args:
            name: Stage identifier string

        Returns:
            The deserialized object

        Raises:
            FileNotFoundError: if no checkpoint exists for name
        """
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found for stage '{name}' at {path}")
        with open(path, "rb") as f:
            value = pickle.load(f)
        log.debug(f"  Checkpoint loaded: {path.name}")
        return value

    def clear(self, name: str) -> None:
        """Delete one checkpoint file."""
        path = self._path(name)
        if path.exists():
            path.unlink()
            log.info(f"  Deleted checkpoint: {path.name}")

    def clear_all(self) -> None:
        """Delete all checkpoint files in the directory."""
        for f in self.directory.glob("checkpoint_*.pkl"):
            f.unlink()
            log.info(f"  Deleted: {f.name}")