import { chatDataPartSchema, fixChatInitialData } from '@/components/chat/chat-stream-state';

it('pass', () => {
  const result = chatDataPartSchema.safeParse(fixChatInitialData({
    'chat': {
      'updated_at': '2024-07-08T10:28:39',
      'id': '019091e3-5cfc-74a3-b5e0-653a73f52af2',
      'title': 'What is TiDB?',
      'engine_id': 1,
      'engine_options': {"llm":{"provider":"openai","openai_chat_model":"gpt-4.1-2025-04-14","gemini_chat_model":"models/gemini-2.0-flash","reranker_provider":"jinaai","reranker_top_k":10,"intent_graph_knowledge":"Given a list of relationships of a knowledge graph as follows. When there is a conflict in meaning between knowledge relationships, the relationship with the higher `weight` and newer `last_modified_at` value takes precedence.\\n\\nKnowledge sub-queries:\\n\\n{% for sub_query, data in sub_queries.items() %}\\n\\nSub-query: {{ sub_query }}\\n\\n  - Entities:\\n\\n{% for entity in data[\'entities\'] %}\\n\\n    - Name: {{ entity.name }}\\n    - Description: {{ entity.description }}\\n\\n{% endfor %}\\n\\n  - Relationships:\\n\\n{% for relationship in data[\'relationships\'] %}\\n\\n    - Description: {{ relationship.rag_description }}\\n    - Last Modified At: {{ relationship.last_modified_at }}\\n    - Meta: {{ relationship.meta | tojson(indent=2) }}\\n\\n{% endfor %}\\n\\n{% endfor %}\\n","normal_graph_knowledge":"Given a list of relationships of a knowledge graph as follows. When there is a conflict in meaning between knowledge relationships, the relationship with the higher `weight` and newer `last_modified_at` value takes precedence.\\n\\n---------------------\\nEntities:\\n\\n{% for entity in entities %}\\n\\n- Name: {{ entity.name }}\\n- Description: {{ entity.description }}\\n\\n{% endfor %}\\n\\n---------------------\\n\\nKnowledge relationships:\\n\\n{% for relationship in relationships %}\\n\\n- Description: {{ relationship.rag_description }}\\n- Weight: {{ relationship.weight }}\\n- Last Modified At: {{ relationship.last_modified_at }}\\n- Meta: {{ relationship.meta | tojson(indent=2) }}\\n\\n{% endfor %}\\n"},"knowledge_graph":{"enabled":true,"depth":2,"include_meta":true,"with_degree":false,"using_intent_search":true}},
      'user_id': null,
      'browser_id': null,
      'created_at': '2024-07-08T10:28:39',
      'deleted_at': null,
      'origin': null,
    },
    'user_message': {
      'id': 60033,
      'created_at': '2024-07-08T10:28:40',
      'role': 'user',
      'trace_url': null,
      'finished_at': null,
      'user_id': null,
      'updated_at': '2024-07-08T10:28:40',
      'ordinal': 1,
      'content': 'What is TiDB?',
      'error': null,
      'sources': [],
      'chat_id': '019091e3-5cfc-74a3-b5e0-653a73f52af2',
      "post_verification_result_url": null,
    },
    'assistant_message': {
      'id': 60034,
      'created_at': '2024-07-08T10:28:40',
      'role': 'assistant',
      'trace_url': 'https://us.cloud.langfuse.com/trace/fd18e8c4-94b7-4b6a-a6e7-3877f07a3d2d',
      'finished_at': '2024-07-08T10:30:12.129128Z',
      'user_id': null,
      'ordinal': 2,
      'content': '### Comprehensive Overview of TiDB\n\n#### What is TiDB?\nTiDB is an open-source distributed SQL database designed to support Hybrid Transactional and Analytical Processing (HTAP) workloads. It is MySQL-compatible and offers horizontal scalability, strong consistency, and high availability[^1][^2]. TiDB aims to provide a one-stop solution for OLTP (Online Transactional Processing), OLAP (Online Analytical Processing), and HTAP services, making it suitable for various use cases that require high availability and strong consistency with large-scale data[^3].\n\n#### Architecture of TiDB\nThe TiDB cluster consists of three main components[^4][^5]:\n1. **TiDB Server**: Handles SQL parsing, query planning, and execution.\n2. **TiKV Server**: Acts as the distributed key-value storage engine, storing the actual data.\n3. **PD (Placement Driver) Server**: Manages cluster metadata, allocates timestamps, and handles data placement and load balancing.\n\nAdditionally, TiDB includes:\n- **TiFlash**: A columnar storage engine for analytical workloads, providing high concurrency for `INSERT` and `UPDATE` operations without impacting OLTP performance[^6].\n- **TiSpark**: A connector that enables Spark to access data stored in TiDB[^7].\n- **TiDB Binlog**: A tool for capturing and replicating data changes[^8].\n- **TiDB Lightning**: A high-performance tool for importing data into TiDB[^9].\n\n#### Key Features of TiDB\n1. **Horizontal Scalability**: TiDB allows for easy horizontal scaling of both computing and storage resources, making it adaptable to changing workloads[^10]. The architecture separates computing from storage, enabling independent scaling[^11].\n2. **High Availability**: TiDB ensures high availability through data replication and the Multi-Raft protocol, guaranteeing data integrity even in the event of failures[^12]. It supports automatic failover when a minority of replicas fail, making it transparent to applications[^13].\n3. **HTAP Capabilities**: TiDB supports both row-based (TiKV) and columnar (TiFlash) storage engines, enabling real-time processing of both transactional and analytical workloads[^14].\n4. **Cloud-Native Design**: TiDB is built for cloud environments, offering flexible scalability, reliability, and security on various cloud platforms[^15]. It integrates seamlessly with Kubernetes and offers a fully-managed service (TiDB Cloud)[^16].\n5. **MySQL Compatibility**: TiDB is compatible with the MySQL 5.7 protocol and ecosystem, allowing for easy migration of applications with minimal code changes[^17]. However, it does not support triggers, stored procedures, and user-defined functions[^18].\n\n#### Ensuring High Availability, Scalability, and Performance\n- **High Availability**: TiDB achieves high availability through its multi-replica architecture and the Multi-Raft protocol, which ensures that data is consistently replicated across multiple nodes[^19]. Transactions can only be committed when data has been successfully written into the majority of replicas[^20].\n- **Scalability**: TiDB\'s architecture allows for flexible and elastic scaling by separating computing from storage. This design enables users to scale out or scale in the computing or storage capacity online as needed[^21].\n- **Performance**: TiDB provides high performance through various optimizations, including the use of TiFlash for analytical workloads and the DeltaTree structure for efficient data modification[^22]. The system also supports distributed transactions using a two-phase commit protocol with optimizations inspired by Google\'s Percolator[^23].\n\n#### Compatibility with MySQL\nTiDB supports most MySQL 5.7 syntax and features, making it highly compatible with MySQL applications[^24]. This compatibility allows users to migrate applications to TiDB without changing a single line of code in many cases[^25]. However, certain features like triggers, stored procedures, and user-defined functions are not supported[^26].\n\n### Conclusion\nTiDB is a robust, scalable, and high-performance distributed SQL database designed for modern data workloads. Its architecture, key features, and compatibility with MySQL make it a versatile solution for various use cases, ensuring high availability, scalability, and performance.\n\n[^1]: [TiDB Overview | PingCAP Docs](https://docs.pingcap.com/tidb/stable/overview)\n[^2]: [TiDB Introduction | PingCAP Docs](https://docs.pingcap.com/tidb/v5.4/overview)\n[^3]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.1/tidb-faq)\n[^4]: [TiDB Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-architecture)\n[^5]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v6.5/tidb-faq)\n[^6]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-faq)\n[^7]: [TiDB Architecture | PingCAP Docs](https://docs.pingcap.com/tidbcloud/tidb-architecture)\n[^8]: [TiDB Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-architecture)\n[^9]: [TiDB Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-architecture)\n[^10]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v6.5/overview)\n[^11]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v5.4/overview)\n[^12]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v7.1/overview)\n[^13]: [TiDB Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-architecture)\n[^14]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v6.5/overview)\n[^15]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v7.1/overview)\n[^16]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v6.5/overview)\n[^17]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v7.1/overview)\n[^18]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-faq)\n[^19]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v5.4/overview)\n[^20]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v6.5/overview)\n[^21]: [TiDB Key Features | PingCAP Docs](https://docs.pingcap.com/tidb/v7.1/overview)\n[^22]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-faq)\n[^23]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v6.5/tidb-faq)\n[^24]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-faq)\n[^25]: [TiDB Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-architecture)\n[^26]: [TiDB Introduction and Architecture | PingCAP Docs](https://docs.pingcap.com/tidb/v7.5/tidb-faq)',
      'error': null,
      "post_verification_result_url": null,
      'sources': [
        {
          'id': 8247,
          'name': 'Overview',
          'source_uri': 'https://docs.pingcap.com/tidb/v7.5/tidb-architecture',
        }, {
          'id': 8600,
          'name': 'TiDB FAQs',
          'source_uri': 'https://docs.pingcap.com/tidb/v7.5/tidb-faq',
        }, {
          'id': 9057,
          'name': 'Overview',
          'source_uri': 'https://docs.pingcap.com/tidbcloud/tidb-architecture',
        }, {
          'id': 9373,
          'name': 'TiDB Introduction',
          'source_uri': 'https://docs.pingcap.com/tidb/v7.1/overview',
        }, {
          'id': 9865,
          'name': 'Overview',
          'source_uri': 'https://docs.pingcap.com/tidb/v7.1/tidb-architecture',
        }, {
          'id': 10191,
          'name': 'TiDB FAQs',
          'source_uri': 'https://docs.pingcap.com/tidb/v7.1/tidb-faq',
        }, {
          'id': 10578,
          'name': 'TiDB Introduction',
          'source_uri': 'https://docs.pingcap.com/tidb/v6.5/overview',
        }, {
          'id': 11370,
          'name': 'TiDB FAQs',
          'source_uri': 'https://docs.pingcap.com/tidb/v6.5/tidb-faq',
        }, {
          'id': 12985,
          'name': 'TiDB Introduction',
          'source_uri': 'https://docs.pingcap.com/tidb/v5.4/overview',
        },
      ],
      'chat_id': '019091e3-5cfc-74a3-b5e0-653a73f52af2',
    },
  } as any));

  if (!result.success) {
    throw result.error;
  }
});