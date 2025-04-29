import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def env():
    print("Loading environment variables")
    load_dotenv()
