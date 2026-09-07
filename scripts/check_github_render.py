#!/usr/bin/env python3
"""Optional official GitHub Markdown HTML audit; sends document text to GitHub.

Requires gh authentication and network access. Stores HTML only at --output.
This verifies GitHub wrappers, not the subsequent browser MathJax/diagram render.
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from validate import ROOT, markdown_files


class InspectHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.math_depth=0
        self.code_depth=0
        self.math_count=0
        self.mermaid_count=0
        self.raw_math=[]
    def handle_starttag(self,tag,attributes):
        attrs=dict(attributes)
        if tag=='math-renderer':
            self.math_depth+=1
            self.math_count+=1
        if tag in ('pre','code'):self.code_depth+=1
        if attrs.get('data-type')=='mermaid':self.mermaid_count+=1
    def handle_endtag(self,tag):
        if tag=='math-renderer':self.math_depth-=1
        if tag in ('pre','code'):self.code_depth-=1
    def handle_data(self,data):
        if not self.math_depth and not self.code_depth and '$$' in data:self.raw_math.append(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in markdown_files():
        payload={'text':path.read_text(),'mode':'gfm','context':'oguzhankir/awesome-video-summarization'}
        response=subprocess.run(['gh','api','--method','POST','markdown','--input','-'],
                                input=json.dumps(payload),text=True,capture_output=True)
        if response.returncode:
            raise SystemExit(f'GitHub render failed for {path.relative_to(ROOT)}: {response.stderr.strip()}')
        relative=path.relative_to(ROOT)
        destination=args.output/(str(relative).replace('/','__')+'.html')
        destination.write_text(response.stdout)
        inspector=InspectHTML();inspector.feed(response.stdout)
        row={'file':str(relative),'math_wrappers':inspector.math_count,
             'mermaid_wrappers':inspector.mermaid_count,'unwrapped_display_delimiters':len(inspector.raw_math)}
        results.append(row)
        if inspector.raw_math:print('REVIEW unwrapped display delimiters:',relative,flush=True)
    (args.output/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'GitHub HTML: {len(results)} documents; {sum(r["math_wrappers"] for r in results)} math wrappers; {sum(r["mermaid_wrappers"] for r in results)} Mermaid wrappers; {sum(r["unwrapped_display_delimiters"] for r in results)} unwrapped display delimiters')
    print('LIMIT: browser client rendering remains a separate Preview check.')
    raise SystemExit(any(r['unwrapped_display_delimiters'] for r in results))


if __name__=='__main__':main()
