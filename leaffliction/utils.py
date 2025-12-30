from __future__ import annotations

from pathlib import Path
from typing import List
import hashlib
import zipfile


class PathManager:
    """
    Centralizes all 'business logic' rules related to paths:
    - directory creation
    - suffix conventions (_Flip, _Rotate, etc.)
    - image iteration
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def ft_ensure_dir(self, path: Path) -> Path:
        """
        Ensure the directory exists, creating it if needed,
        the return the same Path.
        Creates parent directories; no error if it already exist

        Args:
            path (Path): The path from which we should create a folder

        Returns:
            Path: The input path object
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ft_make_suffixed_path(self, image_path: Path, suffix: str) -> Path:
        """
        Add a suffix to a the filename of the path.

        Args:
            image_path (Path): File path to the image
            suffix (str): The suffix to append to the path

        Returns:
            Path: The new path.

        Example:
            input:  /a/b/image1.JPG  + suffix="Flip"
            output: /a/b/image1_Flip.JPG
        """
        new_name = f"{image_path.stem}{str}{image_path.suffix}"
        new_path = image_path.with_name(new_name)
        image_path.rename(new_path)
        return image_path

    def ft_iter_images(
            self,
            root: Path,
            recursive: bool = False
            ) -> List[Path]:
        """
        List images in the given directory (recursive option).

        Args:
            root (Path): the root directory of the listing
            recursive (bool): recursive option

        Returns:
            List[Path]: The list of all images.
        """
        pattern = "**/*" if recursive else "*"
        return [
            file for file in root.glob(pattern)
            if file.is_file() and file.suffix in self.IMAGE_EXTS
        ]


class Hasher:
    """Hash tools (SHA1) for signature.txt."""

    def ft_sha1_file(
            self,
            path: Path,
            chunk_size: int = 1024 * 1024
            ) -> str:
        """
        Returns the SHA1 hex of a path.

        Args:
            path (Path): path to the file/directory to process
            chunk_size (int): Size of chunks to read at once (bytes)

        Returns:
            str: The SHA1 hex of the path
        """
        sha1_hash = hashlib.sha1()

        with open(path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha1_hash.update(chunk)

        return sha1_hash.hexdigest()


class ZipPackager:
    """Compression ZIP (dossier -> zip)."""

    def ft_zip_dir(self, src_dir: Path, out_zip: Path) -> None:
        """
        Zip tout le contenu de src_dir dans out_zip.
        Attention: out_zip ne doit pas être dans src_dir sinon boucle.
        """
        raise NotImplementedError
