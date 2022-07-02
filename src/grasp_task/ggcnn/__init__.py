def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn_model import GGCNN
        return GGCNN
    else:
        raise NotImplementedError("model {} is not implemented".format(network_name))