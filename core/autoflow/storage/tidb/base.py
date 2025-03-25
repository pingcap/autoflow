from sqlalchemy.orm import registry

default_registry = registry()

Base = default_registry.generate_base()
