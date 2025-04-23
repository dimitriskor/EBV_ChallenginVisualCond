import os 

names_clear = os.listdir('/mnt/raid0a/Dimitris/DDD20/clear/')
names_night = os.listdir('/mnt/raid0a/Dimitris/DDD20/night/')
names_glare = os.listdir('/mnt/raid0a/Dimitris/DDD20/glare/')


for clear in names_clear:
    if clear+'.exported.hdf5' in names_clear or '.exported.hdf5' in clear:
        continue
    print(f'Copying {clear}')
    os.system(f'cp /mnt/raid0a/Dimitris/DDD20/clear/{clear} .')
    print(f'Exporting {clear}')
    os.system(f'python3 ddd20-utils/export_ddd20_hdf.py {clear} --display 0')
    os.system(f'rm {clear}')

print(f'Moving clear')
os.system(f'sudo mv *.exported.hdf5 /mnt/raid0a/Dimitris/DDD20/clear/')

for night in names_night:
    if night+'.exported.hdf5' in names_night or '.exported.hdf5' in night:
        continue
    print(f'\n\nCopying {night}')
    os.system(f'cp /mnt/raid0a/Dimitris/DDD20/night/{night} .')
    print(f'Exporting {night}')
    os.system(f'python3 ddd20-utils/export_ddd20_hdf.py {night} --display 0')
    os.system(f'rm {night}')

print(f'Moving night')
os.system(f'sudo mv *.exported.hdf5 /mnt/raid0a/Dimitris/DDD20/night/')


for glare in names_glare:
    if glare+'.exported.hdf5' in names_glare or '.exported.hdf5' in glare:
        continue
    print(f'\n\nCopying {glare}')
    os.system(f'cp /mnt/raid0a/Dimitris/DDD20/glare/{glare} .')
    print(f'Exporting {glare}')
    os.system(f'python3 ddd20-utils/export_ddd20_hdf.py {glare} --display 0')
    os.system(f'rm {glare}')

print(f'Moving glare')
os.system(f'sudo mv *.exported.hdf5 /mnt/raid0a/Dimitris/DDD20/glare/')
