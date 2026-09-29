from pathlib import Path

from voice_assistant.config.config_manager import ConfigManager, Config


def test_default_config_created(tmp_path):
    config_path = tmp_path / "config.json"
    manager = ConfigManager(config_path)
    assert config_path.exists()
    assert manager.get().whisper_model == "base"


def test_config_roundtrip(tmp_path):
    config_path = tmp_path / "config.json"
    manager = ConfigManager(config_path)
    manager.config.whisper_model = "small"
    manager.config.max_search_results = 5
    manager.save()

    reloaded = ConfigManager(config_path)
    assert reloaded.get().whisper_model == "small"
    assert reloaded.get().max_search_results == 5


def test_corrupt_config_falls_back_to_default(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{not valid json")
    manager = ConfigManager(config_path)
    assert manager.get().whisper_model == Config.default().whisper_model
