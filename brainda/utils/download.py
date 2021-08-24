# -*- coding: utf-8 -*-
# Downloading utilies
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/07
# License: MIT License
from mne.datasets.utils import _get_path, _do_path_update
from mne.utils import verbose

import os, shutil
from typing import Union, Optional, Dict
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from pooch import file_hash, retrieve, HTTPDownloader, FTPDownloader


def _url_to_local_path(url: str, local_path: Union[str, Path]) -> str:
    """Mirror a url path in a local destination with keeping folder structure.

    Parameters
    ----------
    url : str
        target file url
    local_path : Union[str, Path]
        local folder to mirror the target file url

    Returns
    -------
    str
        the local path of the target file

    Raises
    ------
    ValueError
        raise ValueError if url is not valid
    """    
    destination = urlparse(url).path
    # First char should be '/', and it needs to be discarded
    if len(destination) < 2 or destination[0] != '/':
        raise ValueError('Invalid URL')
    destination = os.path.join(local_path, url2pathname(destination)[1:])
    return destination

def _get_http(url: str, file_name: Union[str, Path], 
        proxies: Optional[Dict[str, str]] = None, 
        known_hash: Optional[str] = None, **kwargs):
        retrieve(url, known_hash,
            fname=os.path.basename(file_name),
            path=os.path.dirname(file_name),
            downloader=HTTPDownloader(
                progressbar=True, 
                proxies=proxies,
                allow_redirects=True, **kwargs))

def _get_ftp(url: str, file_name: Union[str, Path], 
        known_hash: Optional[str] = None, **kwargs):
        retrieve(url, known_hash,
            fname=os.path.basename(file_name),
            path=os.path.dirname(file_name),
            downloader=FTPDownloader(
                progressbar=True, **kwargs))

def _get_file(url: str, file_name: Union[str, Path], 
        known_hash: Optional[str] = None, **kwargs):
        src_file = urlparse(url).path
        path = os.path.dirname(file_name)
        if not os.path.exists(file_name):
            os.makedirs(path, exist_ok=True)
            shutil.copy(src_file, file_name)

def _fetch_file(url: str, file_name: Union[str, Path], 
        proxies: Optional[Dict[str, str]] = None,
        known_hash: Optional[str]=None, **kwargs):
    scheme = urlparse(url).scheme

    if scheme in ('http', 'https'):
        _get_http(url, file_name, proxies=proxies, known_hash=known_hash, **kwargs)
    elif scheme in ('ftp'):
        _get_ftp(url, file_name, known_hash=known_hash, **kwargs)
    elif scheme in ('file'):
        _get_file(url, file_name, known_hash=known_hash, **kwargs)
    else:
        raise NotImplementedError('Cannot use scheme {:s}'.format(scheme))

@verbose
def mne_data_path(url: str, sign: str, 
        path: Union[str, Path] = None,
        proxies: Optional[Dict[str, str]] = None, 
        force_update: bool = False, 
        update_path: bool = True,
        verbose: Optional[Union[bool, str, int]] = None, **kwargs) -> str:
    """Get the local path of the target file.

    This function returns the local path of the target file, downloading it if needed or requested. The local path keeps the same structure as the url.

    Parameters
    ----------
    url : str
        url of the target file.
    sign : str
        the unique identifier to which the file belongs
    path : Union[str, Path], optional
        local folder to save the file, by default None
    proxies : Optional[Dict[str, str]], optional
        use proxies to download files, e.g. {'https': 'socks5://127.0.0.1:1080'}, by default None
    force_update : bool, optional
        whether to re-download the file, by default False
    update_path : bool, optional
        whether to update mne config, by default True
    verbose : Optional[Union[bool, str, int]], optional
        [description], by default None

    Returns
    -------
    str
        local path of the target file
    """
    sign = sign.upper()
    key = 'MNE_DATASETS_{:s}_PATH'.format(sign)
    key_dest = 'MNE-{:s}-data'.format(sign.lower())
    path = _get_path(path, key, sign)
    destination = _url_to_local_path(url, os.path.join(path, key_dest))
    # Fetch the file
    # forget hash check
    known_hash = None
    if not os.path.exists(destination) or force_update:
        if not os.path.isdir(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        if os.path.isfile(destination):
            os.remove(destination)
    #     known_hash = None
    # else:
    #     known_hash = file_hash(destination)
    
    _fetch_file(
        url, destination,
        proxies=proxies,
        known_hash=known_hash,
        **kwargs)

    _do_path_update(path, update_path, key, sign)
    return destination

