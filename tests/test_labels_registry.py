import os

from scripts.generate_labels_registry import build_labels_registry


def test_labels_registry_structure(tmp_path):
    output_json = tmp_path / "labels.json"
    registry = build_labels_registry(output_path=str(output_json))
    
    assert registry["total_classes"] == 28
    assert len(registry["classes"]) == 28
    
    first = registry["classes"][0]
    assert "id" in first
    assert "slug" in first
    assert "name_id" in first
    assert "name_en" in first
    assert "region" in first
    assert "island" in first
    assert "description_id" in first
    assert "description_en" in first
    assert isinstance(first["is_weak_class"], bool)
    assert os.path.exists(output_json)
    
    # Check that weak classes are correctly flagged
    weak_slugs = {c["slug"] for c in registry["classes"] if c["is_weak_class"]}
    assert weak_slugs == {"Priangan_Merak_Ngibing", "Sogan", "Lasem"}
