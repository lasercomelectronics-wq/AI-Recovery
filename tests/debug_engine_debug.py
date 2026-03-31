from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from blackops.carving.images.engine import ImageCarvingEngine
from blackops.carving.images.base import ImageFormat

def create_mock_png():
    magic = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
    ihdr_data = b'\x00\x00\x00\x0D' + b'IHDR' + b'\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00' + b'\x90\x77\x53\xDE'
    iend_data = b'\x00\x00\x00\x00' + b'IEND' + b'\xAE\x42\x60\x82'
    return magic + ihdr_data + iend_data

engine = ImageCarvingEngine()
png = create_mock_png()
jpg = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xFF\xD9'

data = png + b'\x00' * 100 + jpg

print(f"Data length: {len(data)}")

# Indexer scan
entries = engine.indexer.scan(memoryview(data))
print("Indexer entries:")
for e in entries:
    print(f" - format={e.format} offset={e.offset} confidence={e.confidence}")

# Run carve and show validation results
print('\nCarve results:')
results = list(engine.carve(memoryview(data), 0))
for r in results:
    print(f" - format={r.format} offset_in_block={r.offset_in_block} score={r.score} total_size={r.total_size} integrity_complete={r.integrity_complete}")

print('\nDone')
