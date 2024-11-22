from typing import Any, Union
from pathlib import Path
import io
import lmdb
import pickle
from PIL import Image, ImageFile
import shutil
from time import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _ascii_encode(data: str) -> bytes:
    return data.encode("ascii")

def _pickle_encode(data: Any, protocol: int) -> bytes:
    return pickle.dumps(data, protocol=protocol)

def _pickle_decode(data: bytes) -> Any:
    return pickle.loads(data)


class Database(object):
    _database = None
    _protocol = None
    _length = None
    _has_fetched = False

    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = False,
    ):
        """
        Base class for LMDB-backed databases.

        :param path: Path to the database.
        :param readahead: Enables the filesystem readahead mechanism. Useful only if your database fits in RAM.
        """
        if not isinstance(path, str):
            path = str(path)

        self.path = path
        self.readahead = readahead

    @property
    def database(self):
        if self._database is None:
            self._database = lmdb.open(
                path=self.path,
                readonly=True,
                readahead=self.readahead,
                max_spare_txns=256,
                lock=False,
            )
        return self._database

    @database.deleter
    def database(self):
        if self._database is not None:
            self._database.close()
            self._database = None

    @property
    def protocol(self):
        """
        Read the pickle protocol contained in the database.

        :return: The pickle protocol.
        """
        if self._protocol is None:
            self._protocol = self._get(
                key="protocol",
                fencode=_ascii_encode,
                fdecode=_pickle_decode,
            )
        return self._protocol

    @property
    def keys(self):
        """
        Read the keys contained in the database.

        :return: The set of available keys.
        """
        protocol = self.protocol
        keys = self._get(
            key="keys",
            fencode=lambda key: _pickle_encode(key, protocol=protocol),
            fdecode=_pickle_decode,
        )
        return keys

    def __len__(self):
        """
        Returns the number of keys available in the database.

        :return: The number of keys.
        """
        if self._length is None:
            self._length = len(self.keys)
        return self._length

    def __getitem__(self, item):
        """
        Retrieves an item or a list of items from the database.

        :param item: A key or a list of keys.
        :return: A value or a list of values.
        """
        self._has_fetched = True
        if not isinstance(item, list):
            item = self._get(
                key=item,
                fencode=self._fencode,
                fdecode=self._fdecode,
            )
        else:
            item = self._gets(
                keys=item,
                fencodes=self._fencodes,
                fdecodes=self._fdecodes,
            )
        return item

    def _get(self, key, fencode, fdecode):
        """
        Instantiates a transaction and its associated cursor to fetch an item.

        :param key: A key.
        :param fencode:
        :param fdecode:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                key = fencode(key)
                value = cursor.get(key)
                value = fdecode(value)
        self._keep_database()
        return value

    def _gets(self, keys, fencodes, fdecodes):
        """
        Instantiates a transaction and its associated cursor to fetch a list of items.

        :param keys: A list of keys.
        :param fencodes:
        :param fdecodes:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                keys = fencodes(keys)
                _, values = list(zip(*cursor.getmulti(keys)))
                values = fdecodes(values)
        self._keep_database()
        return values

    def _fencode(self, key: Any) -> bytes:
        """
        Converts a key into a byte key.

        :param key: A key.
        :return: A byte key.
        """
        return _pickle_encode(data=key, protocol=self.protocol)

    def _fencodes(self, keys: list[Any]) -> list[bytes]:
        """
        Converts keys into byte keys.

        :param keys: A list of keys.
        :return: A list of byte keys.
        """
        return [self._fencode(key=key) for key in keys]

    def _fdecode(self, value: bytes) -> Any:
        """
        Converts a byte value back into a value.

        :param value: A byte value.
        :return: A value
        """
        return _pickle_decode(data=value)

    def _fdecodes(self, values: list[bytes]) -> list[Any]:
        """
        Converts bytes values back into values.

        :param values: A list of byte values.
        :return: A list of values.
        """
        return [self._fdecode(value=value) for value in values]

    def _keep_database(self):
        """
        Checks if the database must be deleted.

        :return:
        """
        if not self._has_fetched:
            del self.database

    def __iter__(self):
        """
        Provides an iterator over the keys when iterating over the database.

        :return: An iterator on the keys.
        """
        return iter(self.keys)

    def __del__(self):
        """
        Closes the database properly.
        """
        del self.database


class ImageDatabase(Database):
    def _fdecode(self, value: bytes):
        value = io.BytesIO(value)
        image = Image.open(value)
        return image


def write_database(src_path: Path, database: Path):

    # Customize the how images are to be found and organise them as a dictionary.
    # Here it's just a recursive glob over the whole source image directory.
    d = {image_path.stem: image_path for image_path in sorted(src_path.rglob(f"*jpg"))}

    # Remove any existing database.
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        shutil.rmtree(database)

    # For condor usage, we create a local database on the disk.
    tmp_dir = Path("/tmp") / f"TEMP_{time()}"
    tmp_dir.mkdir(parents=True)

    tmp_database = tmp_dir / f"{database.name}"

    # Create the database.
    with lmdb.open(path=f"{tmp_database}", map_size=2**40) as env:
        # Add the protocol to the database.
        with env.begin(write=True) as txn:
            key = "protocol".encode("ascii")
            value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
            txn.put(key=key, value=value, dupdata=False)
        # Add the keys to the database.
        with env.begin(write=True) as txn:
            key = pickle.dumps("keys")
            value = pickle.dumps(sorted(d.keys()))
            txn.put(key=key, value=value, dupdata=False)
        # Add the images to the database.
        for key, value in sorted(d.items()):
            with env.begin(write=True) as txn:
                with value.open("rb") as file:
                    key = pickle.dumps(key)
                    txn.put(key=key, value=file.read(), dupdata=False)

    # Move the database to its destination.
    shutil.move(f"{tmp_database}", database)

    # Remove the temporary directories.
    shutil.rmtree(tmp_dir)

