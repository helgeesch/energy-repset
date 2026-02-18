"""Tests for Workflow."""
import pytest

from energy_repset.workflow import Workflow


class TestSaveLoad:

    def test_save_raises_not_implemented(self, tmp_path):
        from unittest.mock import MagicMock
        wf = Workflow(
            feature_engineer=MagicMock(),
            search_algorithm=MagicMock(),
            representation_model=MagicMock(),
        )
        with pytest.raises(NotImplementedError, match="serialization"):
            wf.save(tmp_path / "wf.json")

    def test_load_raises_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError, match="deserialization"):
            Workflow.load(tmp_path / "wf.json")
