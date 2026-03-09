"""Prompt template management.

Load, validate, and render prompt templates with variable substitution.
Supports file-based .prompt templates and Jinja2-style {{ var }} syntax.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptTemplate:
    """A reusable prompt template with variable substitution.

    Usage:
        template = PromptTemplate(
            name="analyze",
            text="Analyze the following {{topic}} data: {{data}}",
        )
        rendered = template.render(topic="sales", data="Q1 report")
    """

    name: str
    text: str
    defaults: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str | None = None

    @property
    def variables(self) -> list[str]:
        """Extract all variable names from the template."""
        return re.findall(r"\{\{\s*(\w+)\s*\}\}", self.text)

    def render(self, **kwargs: str) -> str:
        """Render the template with given variables."""
        merged = {**self.defaults, **kwargs}
        missing: list[str] = []
        pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name in merged:
                return merged[var_name]
            missing.append(var_name)
            return match.group(0)

        result = pattern.sub(replacer, self.text)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return result

    @property
    def content_hash(self) -> str:
        """Stable hash of the template text for versioning/audit."""
        return hashlib.sha256(self.text.encode()).hexdigest()

    def validate(self) -> list[str]:
        """Validate template, returning list of issues (empty if valid)."""
        issues = []
        # Check for unmatched braces
        open_count = self.text.count("{{")
        close_count = self.text.count("}}")
        if open_count != close_count:
            issues.append(f"Unmatched braces: {open_count} opening, {close_count} closing")
        # Check that all variables have valid names
        for var in self.variables:
            if not var.isidentifier():
                issues.append(f"Invalid variable name: {var!r}")
        return issues


def load_prompt(path: str | Path) -> PromptTemplate:
    """Load a prompt template from a .prompt file.

    File format:
        ---
        name: analyze
        defaults:
          format: markdown
        ---
        Analyze the following {{topic}} data: {{data}}
        Output in {{format}} format.
    """
    path = Path(path)
    content = path.read_text()

    name = path.stem
    version: str | None = None
    defaults: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    text = content

    # Parse optional YAML frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            text = parts[2].strip()
            try:
                import yaml

                fm = yaml.safe_load(frontmatter) or {}
                if isinstance(fm, dict):
                    name = fm.get("name", name)
                    defaults = fm.get("defaults", {}) or {}
                    version = fm.get("version")
                    metadata = {
                        k: v for k, v in fm.items() if k not in ("name", "defaults", "version")
                    }
            except Exception:
                logger.warning(
                    "PyYAML not installed; using simplified prompt file parser. "
                    "Complex frontmatter (multi-line values, nested structures) "
                    "may not parse correctly. "
                    "Install PyYAML for full support: pip install pyyaml"
                )
                current_section = None
                for line in frontmatter.splitlines():
                    if not line.strip():
                        continue
                    if line.strip().endswith(":") and not line.startswith((" ", "\t")):
                        key = line.split(":", 1)[0].strip()
                        current_section = key
                        continue
                    if ":" in line:
                        key, val = line.split(":", 1)
                        key = key.strip()
                        val = val.strip()
                        if current_section == "defaults" or line.startswith((" ", "\t")):
                            defaults[key] = val
                        elif key == "name":
                            name = val
                        elif key == "version":
                            version = val
                        else:
                            metadata[key] = val

    return PromptTemplate(
        name=name,
        text=text,
        defaults=defaults,
        metadata=metadata,
        version=version,
    )


def load_prompts(directory: str | Path) -> dict[str, PromptTemplate]:
    """Load all .prompt files from a directory."""
    directory = Path(directory)
    templates = {}
    for path in sorted(directory.glob("*.prompt")):
        template = load_prompt(path)
        templates[template.name] = template
    return templates
