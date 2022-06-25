import os, json, sys

with open(sys.argv[1], 'r') as fp:
  data = json.load(fp)

os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
json.dump({'data': data}, open(sys.argv[2], 'w',encoding='utf-8'), indent=2, ensure_ascii=False)