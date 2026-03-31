import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from blackops.carving.images.engine import ImageCarvingEngine
from blackops.carving.images.base import ImageFormat

def create_mock_png():
    # 8-byte magic + 25-byte IHDR (len 13 + type 4 + data 13 + crc 4) + IEND (12 bytes)
    magic = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
    ihdr_data = b'\x00\x00\x00\x0D' + b'IHDR' + b'\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00' + b'\x90\x77\x53\xDE'
    iend_data = b'\x00\x00\x00\x00' + b'IEND' + b'\xAE\x42\x60\x82'
    return magic + ihdr_data + iend_data

def test_byte_shifting():
    engine = ImageCarvingEngine()
    data = b'\x00\x00\x00' + create_mock_png() # Shifted by 3 bytes
    results = [r for r in engine.carve(memoryview(data), 0) if r.score >= 40]
    print(f"Results found (score >= 40): {len(results)}")
    for r in results:
        print(f" - Found {r.format} at {r.offset_in_block} with score {r.score}")
    assert len(results) == 1
    assert results[0].format == ImageFormat.PNG
    assert results[0].offset_in_block == 3
    print("OK Byte shifting test passed")

def test_header_truncation():
    engine = ImageCarvingEngine()
    data = create_mock_png()[:20] # Truncated
    results = list(engine.carve(memoryview(data), 0))
    if results:
        assert results[0].score < 100
        assert results[0].integrity_complete == False
    print("OK Header truncation test passed")

def test_multiple_formats():
    engine = ImageCarvingEngine()
    png = create_mock_png()
    jpg = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xFF\xD9'
    data = png + b'\x00' * 100 + jpg
    results = list(engine.carve(memoryview(data), 0))
    assert len(results) == 2
    assert results[0].format == ImageFormat.PNG
    assert results[1].format == ImageFormat.JPEG
    print("OK Multiple formats test passed")

if __name__ == "__main__":
    print("Starting Forensic Carving Fuzz Tests...")
    test_byte_shifting()
    test_header_truncation()
    test_multiple_formats()
    print("All tests passed!")
