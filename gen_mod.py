import random
import os

def gen_ran(lst, pairs_filepath):
    with open(pairs_filepath, "w") as f:
        f.write("Generated Pairs:\n")

    for x in range(len(lst)):
        if x%2 == 0:
            pth1 = random.choice(lst)
            pth2 = random.choice(lst)

            with open(pairs_filepath, "a") as f:
                if os.path.split(pth1)[0] == os.path.split(pth2)[0]:
                    f.write(f"{pth1}\t{pth2}\ttrue\n")
                else:
                    f.write(f"{pth1}\t{pth2}\tfalse\n")

        else:
            pth1 = random.choice(lst)
            rund = random.choice([-1,1])
            try:
                pth2 = lst[lst.index(pth1)+rund]
            except IndexError:
                pth2 = lst[lst.index(pth1)-1]

            with open(pairs_filepath, "a") as f:
                if os.path.split(pth1)[0] == os.path.split(pth2)[0]:
                    f.write(f"{pth1}\t{pth2}\ttrue\n")
                else:
                    f.write(f"{pth1}\t{pth2}\tfalse\n")