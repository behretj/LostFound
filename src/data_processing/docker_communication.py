"""
Util functions for communicating with (docker) servers.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from contextlib import ExitStack
from typing import Any, Callable

import numpy as np

import open3d as o3d
import requests


class UnsupportedFileFormatException(Exception):
    pass


def _get_content(response: requests.Response, save_path: bytes | str) -> dict:
    """
    Given a requests response, extract the files from the associated zip file
    :param response: the requests response
    :param save_path: where to save files after extracting
    :return: dict of file names and the associated content
    """
    # create folder and extract zip file there
    os.makedirs(save_path, exist_ok=True)
    zip_buffer = io.BytesIO(response.content)
    contents = {}
    with zipfile.ZipFile(zip_buffer, "r") as zipf:
        filenames = [_zipfile.filename for _zipfile in zipf.filelist]
        filenames.sort()
        for filename in filenames:
            # extract content and read content depending on extension
            file = zipf.extract(filename, save_path)
            name, extension = os.path.splitext(filename)
            if extension == ".npy":
                content = np.load(file)
            elif extension == ".ply":
                content = o3d.io.read_triangle_mesh(file)
            elif extension == ".json":
                with open(file, "r") as f:
                    content = json.load(f)
            else:
                raise UnsupportedFileFormatException(extension)
            contents[name] = content
    return contents


def save_files(
    data: list[(str, Callable[[str, Any], Any], Any)], save_path: str
) -> list[str]:
    """
    Save files and return paths so that they can be sent via requests
    :param data: list of names and save functions, and data
    :param save_path: where to save files
    :return: the file paths
    """
    os.makedirs(save_path, exist_ok=True)
    paths = []
    for datum in data:
        name, save_func, value = datum
        path = os.path.join(save_path, name)
        paths.append(path)
        save_func(path, value)

    return paths


def send_request(
    server_address: str,
    paths_dict: dict[str, str],
    params: dict,
    timeout: int,
    save_path: str,
) -> dict[str, Any]:
    """
    Send a request with files to a docker server.
    :param server_address: address of server
    :param paths_dict: paths of files we want to append in the request, consists of {name: path}
    :param params: additional parameters for the request
    :param timeout: timeout for the request
    :param save_path: where to save the files responded with by the server
    """
    with ExitStack() as stack:
        file_dict = {}
        for name, path in paths_dict.items():
            file = stack.enter_context(open(path, "rb"))
            file_dict[name] = file

        response = requests.post(
            server_address,
            files=file_dict,
            params=params,
            timeout=timeout,
        )

    # get returned content
    if response.status_code == 200:  # fail
        contents = _get_content(response, save_path)
    elif response.status_code == 204:  # no content
        contents = {}
    elif response.status_code == 408:  # timeout
        raise TimeoutError("Request timed out!")
    else:
        message = json.loads(response.content)
        print(f"{message['error']}", f"Status code: {response.status_code}", sep="\n")
        raise RuntimeError("Could not get good server response")

    return contents
