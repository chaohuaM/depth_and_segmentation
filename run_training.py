import os.path

from train_model import parse_argument, train_model


def save_options(data, path):
    import yaml
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path.split('/')[-1] + "_config.yaml"
    with open(path + '/' + file_name, encoding='utf-8', mode='w') as f:
        try:
            yaml.dump(data=data, stream=f, allow_unicode=True)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    options = parse_argument()

    save_options(options.__dict__, options.log_dir)
    print("successfully saved!")
    train_model(options)
