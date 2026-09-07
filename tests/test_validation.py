"""Regression checks for evidence parsing and link-audit network boundaries."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_links
import validate


class MarkdownValidationTests(unittest.TestCase):
    def inspect(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.md'
            path.write_text('# Test\n\n' + body)
            return validate.inspect_markdown(path)

    def test_parses_titles_references_and_balanced_parentheses(self):
        errors, visible, _ = self.inspect(
            '[home](README.md "A title")\n\n[paper][ref]\n\n'
            '[ref]: https://doi.org/10.1016/S0031-3203(03)00164-3\n')
        self.assertEqual(errors, [])
        links = [url for _, url in validate.local_links(visible)]
        self.assertIn('README.md', links)
        self.assertIn('https://doi.org/10.1016/S0031-3203(03)00164-3', links)

    def test_link_intent_errors_and_code_exclusion(self):
        for body in ('[broken](missing.md\n', '[broken][undefined]\n'):
            self.assertTrue(self.inspect(body)[0])
        self.assertEqual(self.inspect('`[example](not-a-link`\n')[0], [])
        self.assertEqual(self.inspect('```text\n[example][missing]\n```\n')[0], [])

    def test_raw_display_delimiters_and_fences(self):
        self.assertTrue(self.inspect('\\[\nx\n\\]\n')[0])
        self.assertEqual(self.inspect('```text\n\\[\nx\n\\]\n```\n')[0], [])
        self.assertTrue(self.inspect('```python\nx = 1\n')[0])

    def test_nonfinite_results_rejected_for_every_metric(self):
        for metric in ('F1-Avg', 'Kendall', 'CIDEr'):
            for value in (float('nan'), float('inf'), -float('inf')):
                self.assertTrue(validate.result_number_errors(
                    {'id': 'test', 'metric': metric, 'unit': '%', 'value': value}))
        self.assertTrue(validate.result_number_errors(
            {'id': 'test', 'metric': 'F1-Avg', 'unit': '%', 'value': 101}))
        self.assertEqual(validate.result_number_errors(
            {'id': 'test', 'metric': 'F1-Avg', 'unit': '%', 'value': 42}), [])

    def test_math_operators_cannot_become_setext_headings(self):
        for operator in ('=', '-', '===', '---'):
            errors, _, _ = self.inspect(f'$$\nx\n{operator}\ny\n$$\n')
            self.assertTrue(any('Markdown heading' in error for error in errors))
        self.assertEqual(self.inspect('$$\nx=y\n-z\n$$\n')[0], [])
        self.assertEqual(self.inspect('```math\nx\n=\ny\n```\n')[0], [])

    def test_fenced_math_is_subject_to_macro_policy(self):
        self.assertTrue(self.inspect('```math\n\\operatorname{softmax}(x)\n```\n')[0])
        self.assertEqual(self.inspect('```text\n\\operatorname{softmax}(x)\n```\n')[0], [])

    @unittest.skipUnless(shutil.which('node') and (ROOT/'node_modules').exists(),
                         'npm dependencies not installed')
    def test_github_backtick_math_is_actually_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'node_modules').symlink_to(ROOT/'node_modules', target_is_directory=True)
            shutil.copyfile(ROOT/'scripts/check_render.mjs', root/'scripts/check_render.mjs')
            document = root/'README.md'
            document.write_text('# Test\n\n$`\\thisCommandDoesNotExist`$\n')
            result = subprocess.run(['node', str(root/'scripts/check_render.mjs')],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Undefined control sequence', result.stderr)
            document.write_text('# Test\n\n$`x^2`$\n')
            result = subprocess.run(['node', str(root/'scripts/check_render.mjs')],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('1 math expressions rendered', result.stdout)


class LinkAuditTests(unittest.TestCase):
    def test_known_missing_is_exact_and_still_checked(self):
        issues = [{'url': 'https://example.com/gone', 'status_code':404, 'reason':'Historical archive retained with provenance.'}]
        results = [{'url':'https://example.com/gone','status':'missing','code':404},
                   {'url':'https://example.com/new','status':'missing','code':404}]
        output = audit_links.apply_known_issues(results,issues)
        self.assertEqual([r['status'] for r in output],['known-missing','missing'])
        recovered = audit_links.apply_known_issues([{'url':'https://example.com/gone','status':'ok','code':200}],issues)
        self.assertIn('Recovered',recovered[0]['note'])

    def test_preserves_doi_parentheses_in_registry_prose(self):
        url = 'https://doi.org/10.1016/S0031-3203(03)00164-3'
        self.assertEqual(list(audit_links.prose_urls(f'Primary source ({url}).')), [url])

    def test_credentials_private_ips_and_local_names_are_blocked_before_io(self):
        urls = ['http://127.0.0.1/', 'http://10.0.0.1/',
                'http://169.254.169.254/latest/meta-data/', 'http://[::1]/',
                'http://localhost/', 'http://example.local/',
                'https://user:password@example.com/', 'file:///etc/passwd']
        with patch.object(audit_links, 'public_opener') as build:
            for url in urls:
                self.assertEqual(audit_links.check(url, 1)['status'], 'blocked')
            build.return_value.open.assert_not_called()

    def test_redirects_cannot_bypass_destination_policy(self):
        handler = audit_links.PublicRedirectHandler()
        for url in ['http://127.0.0.1/admin', 'file:///etc/passwd',
                    'https://name:secret@example.com/']:
            with self.assertRaises(audit_links.BlockedTarget):
                handler.redirect_request(Request('https://example.com'), None, 302,
                                         'Found', {}, url)

    def test_private_or_mixed_dns_answers_are_blocked_before_socket_creation(self):
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.168.1.1', 443))
        for candidates in ([private], [public, private]):
            with patch.object(socket, 'getaddrinfo', return_value=candidates), \
                 patch.object(socket, 'socket') as create:
                with self.assertRaises(audit_links.BlockedTarget):
                    audit_links.public_connection(('example.com', 443), 1)
                create.assert_not_called()

    def test_connects_to_validated_ip_without_second_dns_lookup(self):
        info = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))
        with patch.object(socket, 'getaddrinfo', return_value=[info]) as resolve, \
             patch.object(socket, 'socket') as create:
            result = audit_links.public_connection(('example.com', 443), 1)
            self.assertIs(result, create.return_value)
            resolve.assert_called_once()
            create.return_value.connect.assert_called_once_with(('93.184.216.34', 443))

    def test_head_fallback_never_reads_the_response_body(self):
        from urllib.error import HTTPError
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.side_effect = [HTTPError('https://example.com', 405, 'HEAD unsupported', {}, None), response]
        with patch.object(audit_links, 'public_opener', return_value=opener):
            self.assertEqual(audit_links.check('https://example.com', 1)['status'], 'ok')
        self.assertEqual([call.args[0].method for call in opener.open.call_args_list], ['HEAD', 'GET'])
        response.read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
