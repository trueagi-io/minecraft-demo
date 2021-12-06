import shutil
import sys
import os

def main(args):
    dir1 = args[1]
    dir2 = args[2]
    print('merging {0} to {1}'.format(dir2, dir1))
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    files1 = [f for f in files1 if f.startswith('img')]
    files2 = [f for f in files2 if f.startswith('img')]
    ints_max = max([int(f.split('img')[1].split('.png')[0]) for f in files1])
    for i in range(0, len(files2)):
        idx = i + 1
        for kind in ('seg', 'img'):
            target_path = os.path.join(dir1, kind + str(idx + ints_max) + '.png')
            source_path = os.path.join(dir2, files2[i])
            if kind == 'seg':
                source_path = source_path.replace('img', kind)
            print(source_path, ' --> ', target_path)
            shutil.copyfile(source_path, target_path)

if __name__ == '__main__':
    main(sys.argv)

