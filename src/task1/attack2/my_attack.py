from tutorials.craft_adversarial_examples import generate_ae

# parse configurations (into a dictionary) from json file
model_configs = load_from_json(args.model_configs)
data_configs = load_from_json(args.data_configs)
attack_configs = load_from_json(args.attack_configs)

# load the targeted model
model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
target = load_lenet(file=model_file, wrap=True)

# load the benign samples
data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
data_bs = np.load(data_file)
# load the corresponding true labels
label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
labels = np.load(label_file)

# generate adversarial examples for a small subset
data_bs = data_bs[:10]
labels = labels[:10]
generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs)