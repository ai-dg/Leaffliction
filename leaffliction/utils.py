from __future__ import annotations

from pathlib import Path
from typing import List
import hashlib
import shutil


class PathManager:
    """
    Centralizes all 'business logic' rules related to paths:
    - directory creation
    - suffix conventions (_Flip, _Rotate, etc.)
    - image iteration
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def ensure_dir(self, path: Path) -> Path:
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

    def make_suffixed_path(self, image_path: Path, suffix: str) -> Path:
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
        new_name = f"{image_path.stem}{suffix}{image_path.suffix}"
        new_path = image_path.with_name(new_name)
        return new_path

    def mirror_path(
            self,
            src_path: Path,
            src_root: Path,
            target_root: Path
            ) -> Path:
        """
        Given a source file, a source root, and a target root, returns the
        mirror path of the source file respect to the target root.

        Args:
            src_path: Path to the source file to mirror
            src_root: Path to the root directory of the source file
            target_root: Path to the root of the mirror path

        Returns:
            The path to the mirrored source file.
        """
        relative = src_path.resolve().relative_to(src_root.resolve())
        return target_root / relative

    def iter_images(
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
            if file.is_file() and file.suffix.lower() in self.IMAGE_EXTS
        ]


class Hasher:
    """Hash tools (SHA1) for signature.txt."""

    def sha1_file(
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
    """ZIP Compression ZIP (directory -> zip)."""

    def zip_dir(self, src_dir: Path, out_zip: Path) -> None:
        """
        Zip all the content of src_dir into out_zip.

        Args:
            src_dir (Path): the directory to be zipped
            out_zip (Path): the destination path of the zip

        Raises:
            ValueError: If the destination path is inside the source path
        """
        if src_dir.resolve() in out_zip.resolve().parents:
            raise ValueError(
                "Destination path can't be inside the source path."
                )
        shutil.make_archive(
            base_name=str(out_zip.with_suffix('')),
            format='zip',
            root_dir=src_dir
        )
