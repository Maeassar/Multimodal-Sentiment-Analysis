import os

class config:
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir + "/dataset/data/")
    train_data_path = os.path.join(current_dir, 'dataset/train.json')
    test_data_path = os.path.join(current_dir, 'dataset/test.json')
    output_path = os.path.join(current_dir, 'output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    epoch = 20
    learning_rate = 3e-5
    weight_decay = 0
    num_labels = 3

    fuse_model_type = 'SimpleMultimodel'
    only = None

