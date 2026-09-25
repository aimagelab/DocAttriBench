"""
Dataset classes with Pydantic validation for visual source attribution.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetItem(BaseModel):
    """The class for each line of the dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    answer: List[str]
    bbox: List[Tuple[float, float, float, float]]
    type: List[str]
    image_path: str  # The path to the image that contains the evidence to the answer.
    image: Optional[Image.Image] = Field(default=None, exclude=True)
    doc_class: Optional[str] = Field(default=None, exclude=False)


class Dataset(BaseModel):
    """Model for the complete dataset with validation."""

    name: str = Field(..., description="Name of the dataset")
    version: str = Field(default="1.0", description="Version of the dataset")
    description: Optional[str] = Field(None, description="Description of the dataset")
    items: List[DatasetItem] = Field(..., description="List of dataset items")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Dataset-level metadata"
    )
    load_images: bool = Field(
        default=False,
        description="Whether to load a picture from the file when getting the ith item from the dataset.",
    )

    @field_validator("name")
    def validate_name(cls, v):
        """Validate dataset name is not empty."""
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip()

    @classmethod
    def from_list(cls, list_dict: List[Dict]) -> "Dataset":
        """
        Load dataset from a list of dictionaries where each dictionary is structured as a DatasetItem.

        Args:
            list_dict: The list of dictionaries

        Returns:
            Dataset instance
        """
        if not list_dict:
            raise ValueError("The provided list is empty.")

        items = []
        for line_num, line in enumerate(list_dict):
            try:
                item = DatasetItem(**line)
                items.append(item)
            except Exception as e:
                raise ValueError(f"Invalid dataset item on line {line_num}: {e}")

        return cls(
            name="List dataset",
            description=None,
            items=items,
            metadata={
                "source_file": "list",
            },
        )

    @classmethod
    def from_jsonl(cls, file_path: Union[str, Path]) -> "Dataset":
        """
        Load dataset from a JSONL file where each line is a DatasetItem.

        Args:
            file_path: Path to the JSONL file

        Returns:
            Dataset instance
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        items = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    data = json.loads(line)
                    item = DatasetItem(**data)
                    items.append(item)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    raise ValueError(f"Invalid dataset item on line {line_num}: {e}")

        return cls(
            name=file_path.stem,
            description=None,
            items=items,
            metadata={
                "source_file": str(file_path),
            },
        )

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "Dataset":
        """
        Load dataset from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dataset instance
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)

    def to_jsonl(self, file_path: Union[str, Path]) -> None:
        """
        Save dataset items to a JSONL file.

        Args:
            file_path: Path where to save the JSONL file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(
                    json.dumps(item.model_dump(exclude={"image"}), ensure_ascii=False)
                    + "\n"
                )

    def to_json(self, file_path: Union[str, Path]) -> None:
        """
        Save complete dataset to a JSON file.

        Args:
            file_path: Path where to save the JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.model_dump(exclude={"items": {"__all__": {"image"}}})
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def enable_images(self, enable: bool) -> None:
        self.load_images = enable

    def __getitem__(self, index: int) -> DatasetItem:
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Index {index} out of range (0 to {len(self.items)-1})")
        ith_item = self.items[index]
        if self.load_images:
            with Image.open(ith_item.image_path) as im:
                ith_item.image = (
                    im.copy()
                )  # loads into memory; file handle is then closed
        return ith_item

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
