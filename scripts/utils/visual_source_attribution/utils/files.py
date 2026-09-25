import json
from typing import Any, Dict, List, Union


def write_json_file(
    path: str,
    file: Union[List[Any], Dict[Any, Any]],
    verbose: bool = False,
) -> None:
    """
    Saves a JSON file.
    """
    with open(path, "w") as f:
        json.dump(file, f)

    if verbose:
        print(f"Written JSON file to {path}")


def read_json_file(path: str):
    """
    Loads a json file.
    """
    with open(path, "r") as f:
        data = json.load(f)

    return data
