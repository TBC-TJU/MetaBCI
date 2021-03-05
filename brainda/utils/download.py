# -*- coding: utf-8 -*-
# Downloading utilies
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/07
# License: MIT License
from mne.datasets.utils import _get_path, _do_path_update
from mne.utils import verbose

import os, time, shutil
from typing import Union, Optional, Dict
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import requests
from tqdm import tqdm


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
    destination = os.path.join(local_path, destination[1:])
    return destination

def _get_http(url: str, temp_file_name: Union[str, Path], 
        proxies: Optional[Dict[str, str]] = None):
    """Download files to local path via http/https protocols.

    Parameters
    ----------
    url : str
        http/https url of the target file
    temp_file_name : Union[str, Path]
        local path of the target file
    proxies : Optional[Dict[str, str]], optional
        use proxies to download files, e.g. {'https': 'socks5://127.0.0.1:1080'}, by default None
    """
    initial_size = 0
    resume = False

    if os.path.exists(temp_file_name):
        with open(temp_file_name, 'rb', buffering=0) as local_file:
            local_file.seek(0, 2)
            initial_size = local_file.tell()
            del local_file

    # necessary information of the file
    with requests.head(url, proxies=proxies, allow_redirects=True) as response:
        file_size = int(
            response.headers.get('Content-Length', '0').strip()
        )
        # md5_value = response.headers.get('Content-MD5', '').strip()

        if response.headers.get('Accept-Ranges', 'none') == 'bytes':
            resume = True
        url = response.url
        print(url)
    
    if initial_size == file_size:
        # no need redownloading the entire file
        # maybe some md5 check?
        return

    headers = {}
    if resume:
        if initial_size > file_size:
            # last error downloading
            initial_size = 0
        headers['Range'] = 'bytes={:d}-'.format(initial_size)
    else:
        # redownloading
        initial_size = 0

    with requests.get(url, stream=True, proxies=proxies, headers=headers, allow_redirects=True) as response:
        # double-check if support resuming
        content_range = response.headers.get('Content-Range')
        if (content_range is None 
                or not content_range.startswith('bytes {:d}-'.format(initial_size))):
            initial_size = 0

        mode = 'ab' if initial_size > 0 else 'wb'
        chunk_size = 8192
        with open(temp_file_name, mode) as local_file, \
            tqdm(total=file_size, desc=os.path.basename(url), initial=initial_size) as bar:
            while True:
                t0 = time.time()
                chunk = response.raw.read(chunk_size)
                dt = time.time() - t0
                if dt < 5e-2:
                    chunk_size *= 2
                elif dt > 0.5 and chunk_size > 8192:
                    chunk_size = chunk_size//2
                if not chunk:
                    break
                local_file.write(chunk)
                bar.update(len(chunk))

def _get_local(src: str, dst: Union[str, Path]):
    """Copy files from source to destination.

    Parameters
    ----------
    src : str
        source path of the target file
    dst : Union[str, Path]
        destination path of the target file
    """
    if not os.path.isdir(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    shutil.copyfile(src, dst)

def _get_ftp(url: str, temp_file_name: Union[str, Path]):
    """Download files to local path via ftp protocol.

    Parameters
    ----------
    url : str
        ftp url of the target file
    temp_file_name : Union[str, Path]
        local path of the target file
    """
    # use urlopen directly in python3
    # TODO: resuming and proxy features
    req = Request(url)
    mode = 'wb'
    with urlopen(req) as response:
        file_size = int(response.headers.get('Content-Length', '0').strip())
        chunk_size = 8192
        with open(temp_file_name, mode) as local_file, \
            tqdm(total=file_size, desc=os.path.basename(url), initial=0) as bar:
            while True:
                t0 = time.time()
                chunk = response.read(chunk_size)
                dt = time.time() - t0
                if dt < 5e-3:
                    chunk_size *= 2
                elif dt > 0.1 and chunk_size > 8192:
                    chunk_size = chunk_size//2
                if not chunk:
                    break
                local_file.write(chunk)
                bar.update(len(chunk))

def _fetch_file(url: str, file_name: Union[str, Path], 
        proxies: Optional[Dict[str, str]] = None):
    """Fetch files to local path, currently supporting http/https and ftp.

    Parameters
    ----------
    url : str
        url of the target file
    file_name : Union[str, Path]
        local path of the target file
    proxies : Optional[Dict[str, str]], optional
        use proxies to download files, e.g. {'https': 'socks5://127.0.0.1:1080'}, by default None

    Raises
    ------
    NotImplementedError
        raise error if protocol not implemented
    """
    temp_file_name = file_name + '.part'
    scheme = urlparse(url).scheme

    if scheme == '':
        _get_local(url, temp_file_name)
    elif scheme in ('http', 'https'):
        _get_http(url, temp_file_name, proxies=proxies)
    elif scheme in ('ftp'):
        _get_ftp(url, temp_file_name)
    else:
        raise NotImplementedError('Cannot use scheme {:s}'.format(scheme))
            
    shutil.move(temp_file_name, file_name)

@verbose
def mne_data_path(url: str, sign: str, 
        path: Union[str, Path] = None,
        proxies: Optional[Dict[str, str]] = None, 
        force_update: bool = False, 
        update_path: bool = True,
        verbose: Optional[Union[bool, str, int]] = None) -> str:
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
    if not os.path.exists(destination) or force_update:
        if not os.path.isdir(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        if os.path.isfile(destination):
            os.remove(destination)
        _fetch_file(url, destination, proxies=proxies)

    _do_path_update(path, update_path, key, sign)
    return destination

