"""Run the Streamlit app with temporary synthetic local files."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest
from mask_index import SERIES

APP = Path(__file__).resolve().parents[1] / 'app.py'


class ViewerAppTest(unittest.TestCase):
    def run_viewer(self, directory):
        with patch.dict('os.environ', {'DEMO_DATA_DIR': str(directory)}):
            app = AppTest.from_file(str(APP), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_empty_data_shows_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.run_viewer(folder)
            self.assertEqual(len(app.info), 1)
            self.assertEqual(len(app.error), 0)

    def test_complete_case_renders_selector(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for prefix, _ in SERIES:
                Image.new('L', (8, 8), 2).save(directory / f'{prefix}_2.png')
            app = self.run_viewer(directory)
            self.assertEqual(app.selectbox[0].options, ['2'])
            self.assertEqual(len(app.error), 0)

    def test_corrupt_case_renders_error(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for prefix, _ in SERIES:
                Image.new('L', (8, 8), 2).save(directory / f'{prefix}_2.png')
            (directory / 'model_prediction_2.png').write_text('invalid image')
            app = self.run_viewer(directory)
            self.assertEqual(len(app.error), 1)


if __name__ == '__main__':
    unittest.main()
