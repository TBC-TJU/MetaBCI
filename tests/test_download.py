from .base_tmpl import BaseTmpl
import os
from metabci.brainda.utils import download

class TestDownload(BaseTmpl):

    def test_download(self):
        tmpdir = "/tests"

        url = "https://www.google.com/data/folder"
        dest = os.path.join(tmpdir, "data", "folder")
        self.dbgPrint(dest)
        self.assertEqual(dest, download._url_to_local_path(url, tmpdir))

