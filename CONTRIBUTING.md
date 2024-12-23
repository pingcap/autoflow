# How to contribute

## Contributing Guidelines

[pingcap/autoflow](https://github.com/pingcap/autoflow) is an open-source project and we welcome contributions from the community. If you are interested in contributing to the project, please read the following guidelines.

### Before You Get Started

#### Software Prerequisites for Development

In this section, you should have some prerequisites software installed on your local machine:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)
* [Python](https://www.python.org/downloads/)
* [Node.js](https://nodejs.org/en/download/)
* [TiDB Cloud Serverless](https://pingcap.com/ai/?utm_source=tidb.ai&utm_medium=community) or [TiDB Self-Managed](https://www.pingcap.com/tidb-self-managed/?utm_source=tidb.ai&utm_medium=community)

#### Setting up your development environment

Setting up the project on your local machine is the first step to contributing to the project. You can clone the project from the GitHub repository and then start the project on your local machine. You can follow the instructions in the [Deployment Guide](https://tidb.ai/docs/deploy-with-docker) file to set up the project on your local machine.

To test your local changes, you can build and run the project using:

```bash
docker compose -f docker-compose.build.yml up
```

### Your First Contribution

All set to participate in the project? You can start by looking at the [open issues](https://github.com/pingcap/autoflow/issues) in this repo.


### Components of the Project

The project is divided into several components, and you can contribute to any of the following components:
* [Frontend](https://github.com/pingcap/autoflow/tree/main/frontend): The frontend of the project is built using Next.js.
* [Backend](https://github.com/pingcap/tidb.ai/tree/main/backend): The backend of the project is built using FastAPI.
  * [Data Source](https://github.com/pingcap/autoflow/tree/main/backend/app/rag/datasource): The Data Source component is responsible for indexing the data from different type of sources. You can add more data source types to the project.
  * [LLM](https://github.com/pingcap/tidb.ai/tree/main/backend/app/rag/llms): The LLM Engine component is responsible for extracting knowledge from docs and generating responses. You can add more LLM models support to the project.
  * [Reranker](https://github.com/pingcap/tidb.ai/blob/main/backend/app/rag/reranker_model_option.py): The Reranker Engine component is responsible for reranking the results retrieved from the database. You can add more Reranker models support to the project.
  * [Embedding](https://github.com/pingcap/tidb.ai/blob/main/backend/app/rag/embed_model_option.py): The Embedding Engine component is responsible for converting text into vectors. You can add more Embedding models support to the project.
  * [RAG & GraphRAG Engine](https://github.com/pingcap/tidb.ai/tree/main/backend/app/rag): The component is responsible for extracting knowldge from docs and then chunking, indexing and storing the data in the database, also includes retrieving the data from the database and generating the answer for the user.
  * [Documentations](https://github.com/pingcap/tidb.ai/tree/main/frontend/app/src/pages): The documentation of the project is written in Markdown files. You can contribute to the documentation by adding more content to the documentation.]

## Maintainers

Please feel free to reach out to the maintainers if you have any questions or need help with the project.

* [wd0517](https://github.com/wd0517)
* [634750802](https://github.com/634750802)
* [Mini256](https://github.com/Mini256)
* [IANTHEREAL](https://github.com/IANTHEREAL)
* [Cheese](https://github.com/Icemap)

## Discussion

If you have any questions or suggestions, please feel free to open a discussion in the [Discussions](https://github.com/pingcap/tidb.ai/discussions)
