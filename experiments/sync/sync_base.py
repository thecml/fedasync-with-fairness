from plato.clients import simple
from plato.datasources import base
from plato.servers import fedavg
from plato.trainers import basic
import torch
import numpy as np
import yaml

def main():
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    client = simple.Client()
    server = fedavg.Server()
    server.run(client)

if __name__ == "__main__":
    main()
