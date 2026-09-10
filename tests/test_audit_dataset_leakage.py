import os
from PIL import Image
import numpy as np
import pytest
from scripts.audit_dataset_leakage import compute_image_phash, scan_dataset_duplicates

def test_phash_computation(tmp_path):
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img1_path)
    Image.fromarray(arr).save(img2_path)
    
    h1 = compute_image_phash(str(img1_path))
    h2 = compute_image_phash(str(img2_path))
    
    assert isinstance(h1, str)
    assert len(h1) > 0
    assert h1 == h2

def test_scan_dataset_duplicates(tmp_path):
    c1 = tmp_path / "ClassA"
    c2 = tmp_path / "ClassB"
    c1.mkdir()
    c2.mkdir()
    
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_a = c1 / "a.jpg"
    img_b = c2 / "b.jpg"
    Image.fromarray(arr).save(img_a)
    Image.fromarray(arr).save(img_b)
    
    hashes, duplicates = scan_dataset_duplicates(str(tmp_path), threshold=0)
    assert len(duplicates) == 1
    assert duplicates[0]["distance"] == 0
    assert {duplicates[0]["class_a"], duplicates[0]["class_b"]} == {"ClassA", "ClassB"}
