import os
import argparse

def parse_args():
    desc = "Store path to all images in a file"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--root2files', type=str, default=os.path.join('.'),
                        help='Path to files')
    parser.add_argument('--output_root', type=str,
                        help='Path to output directory')
    parser.add_argument('--train_portion', type=float,
                        help='The portion for training images')
    return parser.parse_args()



def read_fv(file):
    fv_id = {}
    with open(file, "r") as f:
        for line in f:
            fv = []
            line = line.split(']')
            line[0] = line[0][1 :]
            line[1] = line[1][2 : -1]
            tmp = line[0].split(",")
            for ele in tmp:
                fv.append(float(ele))
            if line[1] not in fv_id:
                fv_id[line[1]] = [fv]
            else:
                fv_id[line[1]].append(fv)
    return fv_id
    
def write2file(file, data, is_img = False):
    with open(file, "w") as f:
        if(is_img):
            for fv in data:     #for each feature vector
                for i, num in enumerate(fv):
                    if i != (len(fv) - 1):
                        f.write("{},".format(num))
                    else:
                        f.write("{}\n".format(num))
        else:
            for line in data:
                f.write("{}\n".format(line))


def main():
    args = parse_args()
    if args is None:
      exit()
    print("Reading from feature_vector.txt~~\n")
    fv_id = read_fv(os.path.join(args.root2files, "feature_vector.txt"))
    total_id = len(fv_id)

    train_limit = total_id * args.train_portion
    train_imgs = []
    train_ids = []
    test_imgs = []
    test_ids = []
    i = 0
    for idx in fv_id:
        if i < train_limit:
            for ele in fv_id[idx]:
                train_ids.append(idx)
                train_imgs.append(ele)
        else:
            for ele in fv_id[idx]:
                test_ids.append(idx)
                test_imgs.append(ele)
        i += 1
    print("Writing to {}~~\n".format("train_img.csv"))
    write2file(file = os.path.join(args.output_root, "train_img.csv"), data = train_imgs, is_img= True)
    print("Writing to {}~~\n".format("train_id.txt"))
    write2file(file = os.path.join(args.output_root, "train_id.txt"), data = train_ids)
    print("Writing to {}~~\n".format("test_img.csv"))
    write2file(file = os.path.join(args.output_root, "test_img.csv"), data = test_imgs, is_img = True)
    print("Writing to {}~~\n".format("test_id.txt"))
    write2file(file = os.path.join(args.output_root, "test_id.txt"), data = test_ids)

if __name__ == '__main__':
    main()


