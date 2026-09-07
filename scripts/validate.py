#!/usr/bin/env python3
"""Validate registry schemas, foreign keys, Markdown, math and local links."""
import json
import math
from html.parser import HTMLParser
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

import jsonschema
import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from generate_catalog import read_records


def markdown_files(root=ROOT):
    return sorted(p for p in root.rglob('*.md') if not any(part in {'.git','node_modules','.venv','__pycache__'} for part in p.relative_to(root).parts))


def slug(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text).lower()
    text = ''.join(c for c in text if c in '-_ ' or not unicodedata.category(c).startswith(('P', 'S')))
    return text.replace(' ', '-')


def inspect_markdown(path):
    lines = path.read_text().splitlines()
    errors, visible, anchors = [], [], set()
    fence = None
    math_open = False
    details = 0
    previous_heading = 0
    h1 = 0
    seen = {}
    for n, line in enumerate(lines, 1):
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})(.*)$', line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = (token[0], len(token), n, marker[2].strip())
            elif token[0] == fence[0] and len(token) >= fence[1] and not marker[2].strip():
                fence = None
            continue
        if fence:
            if fence[3] == 'math' and re.search(r'\\(?:operatorname|newcommand|def|require)\b', line):
                errors.append(f'{n}: disallowed math macro')
            continue
        visible.append((n, line))
        if line.strip() == '$$':
            if not math_open and n > 1 and lines[n-2].strip():
                errors.append(f'{n}: display math needs preceding blank line')
            if math_open and n < len(lines) and lines[n].strip():
                errors.append(f'{n}: display math needs following blank line')
            math_open = not math_open
            continue
        if re.search(r'(?<!\\)\\[\[\]]', re.sub(r'`+[^`]*`+', '', line)):
            errors.append(f'{n}: use $$ display math instead of raw backslash brackets')
        if re.search(r'\\(?:operatorname|newcommand|def|require)\b', line):
            errors.append(f'{n}: disallowed math macro')
        if math_open:
            if re.fullmatch(r'\s{0,3}(?:=+|-+)\s*', line):
                errors.append(f'{n}: operator-only math line becomes a Markdown heading; join it to an expression')
            continue
        plain = re.sub(r'`+[^`]*`+', '', line)
        if len(re.findall(r'(?<!\\)\$', plain)) % 2:
            errors.append(f'{n}: unbalanced inline math delimiters')
        details += len(re.findall(r'<details(?:\s[^>]*)?>', line)) - line.count('</details>')
        if details < 0:
            errors.append(f'{n}: closing details without opener')
        heading = re.match(r'^(#{1,6})\s+(.+?)(?:\s+#+)?$', line)
        if heading:
            level = len(heading[1])
            h1 += level == 1
            if level > previous_heading + 1:
                errors.append(f'{n}: heading level skips from {previous_heading} to {level}')
            previous_heading = level
            key = slug(heading[2])
            suffix = seen.get(key, 0)
            seen[key] = suffix + 1
            # GitHub suffixes duplicate headings; accidental duplicates still fail.
            if suffix:
                errors.append(f'{n}: duplicate heading anchor {key}')
            anchors.add(key if suffix == 0 else f'{key}-{suffix}')
        anchors.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', line))
    if fence:
        errors.append(f'{fence[2]}: unclosed code fence')
    if math_open:
        errors.append('unclosed display math')
    if details:
        errors.append('unbalanced details blocks')
    if h1 != 1:
        errors.append(f'expected one H1, found {h1}')
    # Compare only true table rows; escaped pipes are part of a cell.
    width = None
    for n, line in visible:
        if line.startswith('|') and line.endswith('|'):
            count = len(re.findall(r'(?<!\\)\|', line))
            if width is not None and count != width:
                errors.append(f'{n}: table column count changed ({width-1} to {count-1})')
            width = count
        else:
            width = None
    errors.extend(markdown_link_errors(visible))
    return errors, visible, anchors


def markdown_tokens(visible):
    # Preserve original line numbers and blank out fenced blocks before parsing.
    numbered = dict(visible)
    source = '\n'.join(numbered.get(n, '') for n in range(1, max(numbered, default=0)+1))
    environment = {}
    tokens = MarkdownIt('commonmark', {'html': True}).enable('table').parse(source, environment)
    return tokens, environment


def walk_tokens(tokens, line=1):
    for token in tokens:
        position = token.map[0]+1 if token.map else line
        yield position, token
        if token.children:
            yield from walk_tokens(token.children, position)


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in {'href', 'src'} and value)


def markdown_link_errors(visible):
    tokens, _ = markdown_tokens(visible)
    errors = []
    for line, token in walk_tokens(tokens):
        if token.type != 'text':
            continue
        # CommonMark leaves malformed links as text. Catch obvious link intent;
        # ordinary bracket notation and inline code remain valid prose.
        prose = re.sub(r'(?<!\\)\$.*?(?<!\\)\$', '', token.content)
        if re.search(r'\[[^\]\n]+\]\(', prose):
            errors.append(f'{line}: malformed inline link destination')
        if re.search(r'\[[^\]\n]+\]\[[^\]\n]*\]', prose):
            errors.append(f'{line}: unresolved reference link')
    return errors


def local_links(visible):
    tokens, environment = markdown_tokens(visible)
    for line, token in walk_tokens(tokens):
        if token.type == 'link_open':
            yield line, token.attrGet('href')
        elif token.type == 'image':
            yield line, token.attrGet('src')
        elif token.type in {'html_inline', 'html_block'}:
            parser = HTMLLinks()
            parser.feed(token.content)
            yield from ((line, link) for link in parser.links)
    # Check definitions even when they are not referenced by visible prose.
    for reference in environment.get('references', {}).values():
        yield reference['map'][0]+1, reference['href']


def result_number_errors(result):
    value = result.get('value')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return [f"{result.get('id', '?')}: result value must be a finite number"]
    metric = result.get('metric', '').lower().replace('_', '-')
    if (metric.startswith('f1') or metric in {'f-score', 'fscore'}) and result.get('unit') == '%' and not 0 <= value <= 100:
        return [f"{result.get('id', '?')}: F1 percentage outside [0, 100]"]
    return []


def validate():
    errors = []
    counts = {}
    records = {}
    for name in ('papers','datasets','results','resources'):
        schema = json.loads((ROOT / f'schemas/{name}.schema.json').read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        try:
            records[name] = read_records(name)
        except (ValueError, TypeError) as error:
            raise SystemExit(f'FAIL: {name}: {error}') from error
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
        for error in validator.iter_errors(records[name]):
            errors.append(f'{name}/{list(error.absolute_path)}: {error.message}')
        if errors:
            raise SystemExit('FAIL: registry schema validation\n' + '\n'.join(errors))
        ids = [r['id'] for r in records[name]]
        if len(ids) != len(set(ids)):
            errors.append(f'{name}: duplicate IDs')
        counts[name] = len(ids)
    paper_ids = {r['id'] for r in records['papers']}
    dataset_ids = {r['id'] for r in records['datasets']}
    for result in records['results']:
        if result['paper_id'] not in paper_ids or result['dataset_id'] not in dataset_ids:
            errors.append(f"{result['id']}: missing paper/dataset foreign key")
        errors.extend(result_number_errors(result))
    for resource in records['resources']:
        if set(resource['paper_ids']) - paper_ids:
            errors.append(f"{resource['id']}: unknown paper_ids")
    documents = {p: inspect_markdown(p) for p in markdown_files()}
    for path, (problems, visible, _) in documents.items():
        errors.extend(f'{path.relative_to(ROOT)}:{p}' for p in problems)
        for line, link in local_links(visible):
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            destination = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not destination.is_relative_to(ROOT):
                errors.append(f'{path.relative_to(ROOT)}:{line}: local link escapes repository: {link}')
            elif not destination.exists():
                errors.append(f'{path.relative_to(ROOT)}:{line}: missing path {link}')
            elif parts.fragment and destination.suffix == '.md':
                anchor_set = documents.get(destination, (None,None,set()))[2]
                if unquote(parts.fragment) not in anchor_set:
                    errors.append(f'{path.relative_to(ROOT)}:{line}: missing anchor {link}')
    issues = json.loads((ROOT / '.github/known-link-issues.json').read_text())
    issue_schema = json.loads((ROOT / 'schemas/link-issues.schema.json').read_text())
    issue_validator = jsonschema.Draft202012Validator(issue_schema, format_checker=jsonschema.FormatChecker())
    for error in issue_validator.iter_errors(issues):
        errors.append('link-issues: ' + error.message)
    if len({issue['url'] for issue in issues}) != len(issues):
        errors.append('link-issues: duplicate URLs')
    yaml.safe_load((ROOT / 'CITATION.cff').read_text())
    workflow_count = 0
    for path in sorted((ROOT / '.github/workflows').glob('*.yml')):
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get('jobs'), dict):
            errors.append(f'{path}: invalid workflow structure')
        workflow_count += 1
    subprocess.run([sys.executable, str(ROOT / 'scripts/generate_catalog.py'), '--check'], check=True)
    if errors:
        print('\n'.join(errors))
        raise SystemExit(f'FAIL: {len(errors)} validation errors')
    print(f'PASS: {len(documents)} Markdown files; local links, anchors, fences, math and tables')
    print('PASS: JSON schemas and foreign keys: ' + ', '.join(f'{v} {k}' for k,v in counts.items()))
    print(f'PASS: {workflow_count} workflow YAML files')


if __name__ == '__main__':
    validate()
