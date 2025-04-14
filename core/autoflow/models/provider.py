from abc import ABC


class ProviderRegistry(ABC):
    def register(self, name: str):
        pass

    def get_provider_credentials(self):
        pass
