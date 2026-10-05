import ast
import __future__
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock


class DownloadDirectoryTests(unittest.TestCase):
    def check_directory(self, custom):
        source = Path(__file__).resolve().parents[1] / 'resemble_enhance/enhancer/download.py'
        tree = ast.parse(source.read_text())
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloader = Mock(side_effect=lambda url, path: Path(path).write_bytes(b'synthetic'))
            ns = {'Path': Path, 'RUN_NAME': 'enhancer_stage2', '__file__': str(root / 'package/enhancer/download.py'), 'torch': types.SimpleNamespace(hub=types.SimpleNamespace(download_url_to_file=downloader))}
            exec(compile(functions, str(source), 'exec', flags=__future__.annotations.compiler_flag), ns)
            requested = root / 'custom' if custom else None
            expected = requested or root / 'package/model_repo/enhancer_stage2'
            actual = ns['download'](requested)
            self.assertEqual(actual, expected)
            self.assertTrue((actual / 'hparams.yaml').is_file())
            self.assertTrue((actual / 'ds/G/default/mp_rank_00_model_states.pt').is_file())
            self.assertEqual(downloader.call_count, 3)
            self.assertEqual(ns['download'](requested), expected)
            self.assertEqual(downloader.call_count, 3)

    def test_custom_directory(self):
        self.check_directory(True)

    def test_default_directory(self):
        self.check_directory(False)


if __name__ == '__main__':
    unittest.main()
