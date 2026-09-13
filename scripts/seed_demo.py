"""Reset or advance the selected synthetic replay through the public API."""
import argparse
import json
import urllib.request


def request(url, method='GET', payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--scenario',default='sleeper_bustout',choices=['sleeper_bustout','legitimate','identity_farm','false_positive'])
    parser.add_argument('--steps',type=int,default=0)
    args=parser.parse_args()
    state=request(f'{args.url}/api/scenarios/{args.scenario}/reset','POST')
    for _ in range(min(args.steps,state['scenario']['total_events'])):
        state=request(f'{args.url}/api/scenarios/{args.scenario}/step','POST')
    print(json.dumps({'scenario':state['scenario'],'accounts':len(state['accounts']),'alerts':len(state['alerts'])},indent=2))
