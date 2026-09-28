"""Regression checks for the original viewer's missing-file failure mode."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from mask_index import SERIES, discover_cases, load_case_images


class DemoFilesTest(unittest.TestCase):
    def test_missing_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            complete, available = discover_cases(Path(folder) / 'missing')
            self.assertEqual(complete, [])
            self.assertTrue(all(not values for values in available.values()))

    def test_partial_case_is_excluded_and_positive_ids_are_sorted(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for case_id in [12, 2]:
                for prefix, _ in SERIES:
                    Image.new('L', (8, 8), case_id).save(directory / f'{prefix}_{case_id}.png')
            Image.new('L', (8, 8)).save(directory / 'ground_truth_3.png')
            (directory / 'ground_truth_0.png').touch()
            (directory / 'unrelated_99.png').touch()
            complete, available = discover_cases(directory)
            self.assertEqual(complete, [2, 12])
            self.assertEqual(available['ground_truth'], {2, 3, 12})
            self.assertEqual(len(load_case_images(directory, 2)), 4)

    def test_corrupt_image_raises_for_ui_to_handle(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for prefix, _ in SERIES:
                (directory / f'{prefix}_1.png').write_text('invalid image')
            with self.assertRaises(OSError):
                load_case_images(directory, 1)


if __name__ == '__main__':
    unittest.main()
