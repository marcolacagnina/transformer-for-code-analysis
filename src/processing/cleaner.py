import logging
import re
import black

logger = logging.getLogger(__name__)


_SINGLEQUOTE_STR_PATTERN = r"\".*?\""
_DOUBLE_QUOTE_STR_PATTERN = r"\'.*?\'"
_SINGLEQUOTE_DOCSTRING_PATTERN = r"^\s*'{3,}[\s\S]*?'{3,}"
_DOUBLEQUOTE_DOCSTRING_PATTERN = r"^\s*\"{3,}[\s\S]*?\"{3,}"
_SINGLELINE_COMMENT_PATTERN = r"#[^\r\n]*$"

_DOCSTRINGS_REGEX = re.compile(f"({_SINGLEQUOTE_DOCSTRING_PATTERN}|{_DOUBLEQUOTE_DOCSTRING_PATTERN})", re.MULTILINE)
_COMMENTS_REGEX = re.compile(f"({_SINGLEQUOTE_STR_PATTERN}|{_DOUBLE_QUOTE_STR_PATTERN})|({_SINGLELINE_COMMENT_PATTERN})", re.MULTILINE)


def _remove_docstrings(code: str) -> str:
    return _DOCSTRINGS_REGEX.sub("", code)


def _comment_replacer(match):
    return match.group(1) if match.group(1) is not None else ""


def _remove_singleline_comments(code: str) -> str:
    return _COMMENTS_REGEX.sub(_comment_replacer, code)


def clean_code(input_code: str):
    if not isinstance(input_code, str):
        return None

    code = _remove_docstrings(input_code)
    code = _remove_singleline_comments(code)
    code = re.sub(r"^__\w+__\s*=.*\n", "", code, flags=re.MULTILINE)

    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        return input_code