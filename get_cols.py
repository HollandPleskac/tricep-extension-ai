def get_cols(num_landmarks):
    cols = []
    for i in range(num_landmarks):
        cols.append(str(i) + "x")
        cols.append(str(i) + "y")
    return cols