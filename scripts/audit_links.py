#!/usr/bin/env python3
"""Public HTTP(S) link audit; response bodies are never read.

Only 404/410 establish missing links. 401/403/429 are restricted. Private
addresses, URL credentials and unsafe redirect destinations are blocked.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import http.client
import ipaddress
import json
from pathlib import Path
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (Request, build_opener, ProxyHandler, HTTPHandler,
                            HTTPSHandler, HTTPRedirectHandler)
from validate import ROOT, markdown_files, inspect_markdown, local_links


class BlockedTarget(ValueError):
    """A link must not be requested from the runner's network."""


def validate_url(url):
    parts = urlsplit(url)
    if parts.scheme not in {'http', 'https'} or not parts.hostname:
        raise BlockedTarget('only absolute HTTP(S) links are permitted')
    if parts.username is not None or parts.password is not None:
        raise BlockedTarget('URL credentials are not permitted')
    if any(ord(char) < 32 for char in url) or '\\' in parts.netloc:
        raise BlockedTarget('invalid URL authority or control character')
    host = parts.hostname.rstrip('.').lower()
    if host == 'localhost' or host.endswith(('.localhost', '.local')):
        raise BlockedTarget('local hostnames are not permitted')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None  # Actual DNS answers are checked at socket creation.
    if address is not None and not address.is_global:
        raise BlockedTarget('non-public IP address is not permitted')
    try:
        port = parts.port
    except ValueError as error:
        raise BlockedTarget(str(error)) from error
    if port is not None and not 1 <= port <= 65535:
        raise BlockedTarget('invalid port')
    return parts


def public_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
                      source_address=None, **kwargs):
    """Validate DNS answers and connect to those exact IPs, avoiding DNS rebinding."""
    host, port = address
    candidates = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not candidates:
        raise OSError('host resolved to no addresses')
    if any(not ipaddress.ip_address(item[4][0].split('%')[0]).is_global for item in candidates):
        raise BlockedTarget('DNS includes a non-public address')
    last_error = None
    for family, kind, protocol, _, sockaddr in candidates:
        connection = socket.socket(family, kind, protocol)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                connection.settimeout(timeout)
            if source_address:
                connection.bind(source_address)
            connection.connect(sockaddr)
            return connection
        except OSError as error:
            last_error = error
            connection.close()
    raise last_error or OSError('connection failed')


class PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = public_connection


class PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = public_connection


class PublicHTTPHandler(HTTPHandler):
    def http_open(self, request):
        validate_url(request.full_url)
        return self.do_open(PublicHTTPConnection, request)


class PublicHTTPSHandler(HTTPSHandler):
    def https_open(self, request):
        validate_url(request.full_url)
        return self.do_open(PublicHTTPSConnection, request, context=self._context)


class PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def public_opener():
    # Environment proxies could resolve a different destination on our behalf.
    return build_opener(ProxyHandler({}), PublicHTTPHandler(), PublicHTTPSHandler(),
                        PublicRedirectHandler())


def prose_urls(value):
    """Retain balanced DOI parentheses, removing only surrounding punctuation."""
    for candidate in re.findall(r'https?://[^\s<>"\']+', value):
        candidate = candidate.rstrip('.,;')
        while candidate.endswith(')') and candidate.count(')') > candidate.count('('):
            candidate = candidate[:-1]
        yield candidate.split('#')[0]


def collect_urls():
    links = set()
    for path in markdown_files():
        for _, url in local_links(inspect_markdown(path)[1]):
            if url.startswith(('https://', 'http://')):
                links.add(url.split('#')[0])

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            links.update(prose_urls(value))

    for path in (ROOT / 'data').rglob('*.json'):
        visit(json.loads(path.read_text()))
    return sorted(links)


def check(url, timeout):
    opener = public_opener()
    # Some publishers reject HEAD; GET responses are closed without reading bodies.
    for method in ('HEAD', 'GET'):
        try:
            validate_url(url)
            request = Request(url, method=method, headers={
                'User-Agent': 'awesome-video-summarization-link-audit/1.0'})
            with opener.open(request, timeout=timeout) as response:
                code = response.status
            return {'url': url, 'status': 'ok', 'code': code}
        except BlockedTarget as error:
            return {'url': url, 'status': 'blocked', 'detail': str(error)}
        except HTTPError as error:
            code = error.code
            error.close()
            if method == 'HEAD' and code in (403, 404, 405, 429, 501):
                continue
            status = ('missing' if code in (404, 410) else
                      'restricted' if code in (401, 403, 429) else 'retry')
            return {'url': url, 'status': status, 'code': code}
        except (URLError, TimeoutError, OSError, ValueError) as error:
            return {'url': url, 'status': 'retry', 'detail': str(error)}


def apply_known_issues(results, issues):
    known = {issue['url']: issue for issue in issues}
    for result in results:
        issue = known.get(result['url'])
        if issue and result['status'] == 'missing' and result.get('code') == issue['status_code']:
            result['status'] = 'known-missing'
            result['note'] = issue['reason']
        elif issue and result['status'] == 'ok':
            result['note'] = 'Recovered: remove the stale known-link-issues entry after review.'
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/avs-link-report.json'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        parser.error('workers, timeout and limit must be positive')
    urls = collect_urls()
    if args.limit:
        urls = urls[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda url: check(url, args.timeout), urls))
    issues = json.loads((ROOT / '.github/known-link-issues.json').read_text())
    results = apply_known_issues(results, issues)
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print('External links: '+', '.join(f'{key}={value}' for key, value in
          sorted(Counter(record['status'] for record in results).items())))
    for record in results:
        if record['status'] in {'missing', 'known-missing', 'blocked'}:
            print(f"{record['status'].upper()} {record.get('code', record.get('detail'))} {record['url']}")
    print(f'Details: {args.output}')
    raise SystemExit(any(record['status'] in {'missing', 'blocked'} for record in results))


if __name__ == '__main__':
    main()
