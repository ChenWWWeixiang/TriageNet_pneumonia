

import requests
for i in range(9):
    url='https://zenodo.org/record/4708800/files/lola11-0{}.mha?download=1'.format(i+1)
    name = '/mnt/data5/lola11-{}.mha'.format(i+1)
    r = requests.get(url)
    with open(name, "wb") as code:
        code.write(r.content)