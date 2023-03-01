from .fedbase import BasicServer, BasicClient

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def iterate(self, t):
        self.selected_clients = self.sample()
        for i in self.selected_clients:
            self.clients_frequency[i] = self.clients_frequency[i] + 1

        # training
        models, train_losses = self.communicate(self.selected_clients)
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        # print(self.client_vols)
        return

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


