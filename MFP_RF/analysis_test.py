if __name__ == '__main__':
    with open('analyze_random_forest_cv_inner_small_aliper.out', 'r') as f:
        lines = f.readlines()

    idx_set = set()
    for line in lines:
        line = line.strip()
        if line == '' or 'Class Num' in line:
            continue
        line = line.split(':')
        idx = int(line[0])
        idx_set.add(idx)

    print(len(idx_set))
