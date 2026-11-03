"""Regression test for the 3-channel OME-TIFF case (Task 12 found that
io_layers._open_layer_tiff returned a column instead of the requested
channel when the channel count was smaller than the image height).

Run from the project root:  python -m unittest tests.test_open_layer_multichannel -v
(also collected by pytest if it is installed).

The synthetic file is written inside tests/_tmp (removed afterwards), never
outside the project.
"""
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import io_layers  # noqa: E402
import prep  # noqa: E402

TMP = Path(__file__).resolve().parent / "_tmp"
NAMES = ["DAPI", "CK8 - TRITC", "VEGF - TRITC"]
H, W = 96, 128  # H and W both much larger than the channel count (3)


def _channel_data():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 4000, size=(H, W)).astype(np.uint16)
    # Distinct channels so a column or a wrong channel cannot match.
    return np.stack([base, (base // 2 + 500).astype(np.uint16), (65535 - base).astype(np.uint16)], axis=0)


class OpenLayerMultichannelTiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TMP.mkdir(exist_ok=True)
        cls.path = TMP / "three_channel.ome.tif"
        cls.data = _channel_data()
        tifffile.imwrite(
            str(cls.path), cls.data, ome=True, tile=(16, 16),
            metadata={"axes": "CYX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0,
                      "PhysicalSizeXUnit": "\u00b5m", "PhysicalSizeYUnit": "\u00b5m",
                      "Channel": {"Name": NAMES}},
        )
        cls.desc = io_layers.describe_input(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_descriptor(self):
        self.assertEqual(self.desc["kind"], "multichannel_fluorescence")
        self.assertEqual(self.desc["channel_names"], NAMES)
        self.assertAlmostEqual(self.desc["pixel_size_um"], 1.0)

    def test_returns_requested_channel_not_a_column(self):
        # Target equals the native size (1 um/px), so no resampling happens
        # and the result must equal the normalized channel plane exactly.
        for idx, name in enumerate(NAMES):
            for selector in (idx, name):
                with self.subTest(channel=selector):
                    img, um = io_layers.open_layer(self.desc, 1.0, channel=selector)
                    self.assertEqual(img.shape, (H, W))
                    self.assertAlmostEqual(um, 1.0)
                    np.testing.assert_array_equal(img, prep.normalize_dapi(self.data[idx]))

    def test_default_channel_is_dapi(self):
        img, _um = io_layers.open_layer(self.desc, 1.0)
        np.testing.assert_array_equal(img, prep.normalize_dapi(self.data[0]))


if __name__ == "__main__":
    unittest.main()
