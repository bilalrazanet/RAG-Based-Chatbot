import importlib.util, json
pkgs = ['tensorflow','kenlm','pyctcdecode']
res = {}
for p in pkgs:
    try:
        res[p] = importlib.util.find_spec(p) is not None
    except Exception as e:
        res[p] = False
with open('check_installs_output.json','w') as f:
    json.dump(res, f)
print('done')
