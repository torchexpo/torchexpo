"""Model Utils"""
from typing import Iterator
import urllib


USER_AGENT = "torchexpo"


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
) -> None:
    with open(destination, "wb") as writer:
        for chunk in content:
            if not chunk:
                continue
            writer.write(chunk)


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT})) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""), filename)
