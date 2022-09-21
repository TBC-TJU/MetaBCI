# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pytest

from brainhacker.utils.download import _url_to_local_path, _fetch_file, mne_data_path

def test_url_to_local_path(tmpdir):
    url = 'https://www.google.com/'
    with pytest.raises(ValueError):
        _url_to_local_path(url, tmpdir)
    url = 'https//www.google.com/data/folder'
    with pytest.raises(ValueError):
        _url_to_local_path(url, tmpdir)
    url = 'https://www.google.com/data/folder'
    dest = os.path.join(tmpdir, 'data', 'folder')
    assert dest == _url_to_local_path(url, tmpdir)




    

        