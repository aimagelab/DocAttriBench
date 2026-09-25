import langid
import nltk
from typing import List, Tuple
import re


nltk.download("punkt_tab", quiet=True)


LANGUAGE_MAP = {
    "en": "english",
    "zh": "chinese",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "ja": "japanese",
}


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect the language of a given text using langid.
    """
    lang, confidence = langid.classify(text)
    if lang in LANGUAGE_MAP:
        lang = LANGUAGE_MAP[lang]
    else:
        lang = "english"
        print("[WARNING] The detected language is not supported.")
    return lang, confidence


def split_text(text: str, lang: str = "en") -> list[str]:
    """
    Split text into sentences using NLTK's punkt tokenizer.
    """
    return nltk.tokenize.sent_tokenize(text, language=lang)


def qwen_2_5_box_extractor(text: str) -> List[List[int]]:
    """
    Extracts boxes from text generated with qwen2.5 and qwen2 models.
    """

    start_tag = "<box>"
    end_tag = "</box>"

    pattern = re.compile(
        re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
        re.DOTALL | re.IGNORECASE,
    )

    # ---- extraction ----

    boxes = list()

    if "<box>" in text:

        for match in pattern.finditer(text):
            content = match.group(1).strip()

            if "," in content:
                coordinates_str = content.split(",")
            else:
                coordinates_str = content.split(" ")

            if len(coordinates_str) != 4:
                continue

            try:
                coordinates = [round(float(x.strip())) for x in coordinates_str]
            except:
                continue

            boxes.append(coordinates)

    return boxes


def internvl_2_5_box_extractor(text: str) -> List[List[int]]:
    """
    Extracts boxes from text generated with qwen2.5 and qwen2 models.
    """

    start_tag = "["
    end_tag = "]"

    pattern = re.compile(
        re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
        re.DOTALL | re.IGNORECASE,
    )

    # ---- extraction ----

    boxes = list()

    if ("[" in text) and ("]" in text):

        for match in pattern.finditer(text):
            content = match.group(1).strip()

            if "," in content:
                coordinates_str = content.split(",")
            else:
                coordinates_str = content.split(" ")

            if len(coordinates_str) != 4:
                continue

            try:
                coordinates = [round(float(x.strip())) for x in coordinates_str]
            except:
                continue

            boxes.append(coordinates)

    return boxes
