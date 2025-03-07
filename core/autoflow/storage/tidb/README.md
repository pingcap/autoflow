# TiDB Vector SDK V2

A powerful Python SDK for vector storage and retrieval operations with TiDB.

- 🔄 Automatic embedding generation
- 🔍 Vector similarity search
- 🎯 Advanced filtering capabilities
- 📦 Bulk operations support

## Installation

```bash
pip install autoflow-ai
# TODO: move to `pip install tidb-vector`
```

## Configuration

Go [tidbcloud.com](http://tidbcloud.com/) to create a free TiDB database cluster

Configuration can be provided through environment variables, or using `.env`:

```dotenv
DATABASE_URL=mysql+pymysql://<username>:<password>@<host>:4000/<database>
OPENAI_API_KEY=sk-proj-****
```

## Quick Start

```python
from autoflow.storage.tidb import TiDBClient
from sqlmodel import Field
from autoflow.llms.embeddings import EmbeddingFunction

# Connect to TiDB
# Format: mysql+pymysql://<>:<password>@<host>:4000/<database>
db = TiDBClient.connect("your_database_url")

# Define your model with auto-embedding
text_embed = EmbeddingFunction("openai/text-embedding-3-small")
class Chunk(TiDBModel, table=True):
    __tablename__ = "chunks"
    id: int = Field(primary_key=True)
    text: str = Field()
    text_vec: Optional[Any] = text_embed.VectorField(source_field="text")
    user_id: int = Field()

# Create table and insert data
table = db.create_table(schema=Chunk)
table.bulk_insert([
    Chunk(id=1, text="The quick brown fox jumps over the lazy dog", user_id=1),
    Chunk(id=2, text="A quick brown dog runs in the park", user_id=2),
    Chunk(id=3, text="The lazy fox sleeps under the tree", user_id=2),
    Chunk(id=4, text="A dog and a fox play in the park", user_id=3)
])

# Search for similar texts
results = table.search("A quick fox in the park").limit(3).to_pydantic()
```

## Detailed Usage

### Connect to TiDB

```python
from autoflow.storage.tidb import TiDBClient

db = TiDBClient.connect("your_database_url")
```

### Create table

```python
from sqlmodel import Field
from autoflow.llms.embeddings import EmbeddingFunction
from autoflow.storage.tidb.constants import DistanceMetric

text_embed = EmbeddingFunction("openai/text-embedding-3-small")

class Chunk(TiDBModel, table=True):
    __tablename__ = "chunks"
    id: int = Field(primary_key=True)
    text: str = Field()
    text_vec: Optional[Any] = text_embed.VectorField(source_field="text")
    user_id: int = Field()

table = db.create_table(schema=Chunk)
```

### Insert data

```python
# Insert single record
table.insert(Chunk(id=1, text="foo", user_id=1))

# Bulk insert multiple records
table.bulk_insert([
    Chunk(id=2, text="bar", user_id=2),
    Chunk(id=3, text="biz", user_id=2),
    Chunk(id=4, text="qux", user_id=3)
])
```

### Query Data

**Get data by ID**

```python
result = table.get(1)
```

**Query data with filters**

```python
results = table.query({
    "user_id": 1
})
```

### Update Data

```python
table.update(
    values={
        "text": "world"
    },
    filters={
        "id": 1
    }
)
```

### Delete Data

```python
# Delete by id
table.delete(filters={"id": 1})

# Delete multiple records
table.delete(filters={"user_id": 2})
```

### Vector Search

```python
# Search with vector
results = (
    table.search([1, 2, 3])
    .distance_metric(metric=DistanceMetric.COSINE)  # Set distance metric
    .num_candidate(20)
    .filter({"user_id": 1})
    .limit(5)
    .to_pydantic()
)

# Search with text
results = table.search("your search text").limit(5).to_pydantic()
```

## Advanced Filtering

TiDB Client supports various filter operators for flexible querying:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal to | `{"field": {"$eq": "hello"}}` |
| `$gt` | Greater than | `{"field": {"$gt": 1}}` |
| `$gte` | Greater than or equal | `{"field": {"$gte": 1}}` |
| `$lt` | Less than | `{"field": {"$lt": 1}}` |
| `$lte` | Less than or equal | `{"field": {"$lte": 1}}` |
| `$in` | In array | `{"field": {"$in": [1, 2, 3]}}` |
| `$nin` | Not in array | `{"field": {"$nin": [1, 2, 3]}}` |
| `$and` | Logical AND | `{"$and": [{"field1": 1}, {"field2": 2}]}` |
| `$or` | Logical OR | `{"$or": [{"field1": 1}, {"field2": 2}]}` |

```python
# Example queries using different operators
table.query({"user_id": 1})  # Implicit $eq
table.query({"id": {"$gt": 1}})  # Greater than
table.query({"id": {"$in": [1, 2, 3]}})  # In array
table.query({
    "$and": [
        {"user_id": 1},
        {"id": {"$gt": 1}}
    ]
})  # Logical AND
```

