import yaml
path = 'configs/config_4.yaml'
res = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)
print(type(res))
