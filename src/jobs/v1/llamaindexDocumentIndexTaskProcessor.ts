import type { DocumentIndexTaskProcessor } from '@/core/services/indexing';
import { DBv1, getDb, tx } from '@/core/v1/db';
import type { Json } from '@/core/v1/db/schema';
import { uuidToBin, vectorToSql } from '@/lib/kysely';
import { getEmbedding } from '@/lib/llamaindex/converters/embedding';
import { getLLM } from '@/lib/llamaindex/converters/llm';
import { fromFlowReaders } from '@/lib/llamaindex/converters/reader';
import { createIndexIngestionPipeline } from '@/lib/llamaindex/indexDocument';
import { baseRegistry } from '@/rag-spec/base';
import { getFlow } from '@/rag-spec/createFlow';
import type { InsertObject } from 'kysely';
import { KeywordExtractor, NodeRelationship, QuestionsAnsweredExtractor, SentenceSplitter, SentenceWindowNodeParser, SummaryExtractor, TitleExtractor } from 'llamaindex';
import type { RelatedNodeType } from 'llamaindex/Node';
import type { UUID } from 'node:crypto';

interface LlamaindexDocumentChunkNodeTable {
  document_id: number;
  embedding: unknown | null;
  hash: string;
  id: Buffer;
  index_id: number;
  metadata: Json;
  text: string;
}

declare module '@/core/v1/db/schema' {
  interface DB extends Record<`llamaindex_document_chunk_node_${string}`, LlamaindexDocumentChunkNodeTable> {
  }
}

export function createLlamaindexDocumentIndexTaskProcessor (): DocumentIndexTaskProcessor {
  return async (task, document, index, mutableInfo) => {
    const flow = await getFlow(baseRegistry, undefined, index.config);

    // Initialize the reader from legacy loaders.
    // TODO: configurable via `index.config`
    const reader = fromFlowReaders(flow); // wrapped llamaindex.reader auto choosing rag.loader

    // Initialize llamaindex node parser from config.
    const { textSplitter, ...parserConfig } = index.config.parser;
    const parser = new SentenceWindowNodeParser({
      textSplitter: new SentenceSplitter(textSplitter),
      ...parserConfig,
    });

    // Select and config the llm for indexing (metadata extractor).
    const llm = getLLM(flow, index.config.llm.provider, index.config.llm.config);
    llm.metadata.model = index.config.llm.config.model;

    // Select and config the embedding (important and immutable)
    const embedding = getEmbedding(flow, index.config.embedding.provider, index.config.embedding.config);

    // Create the default llamaindex pipeline
    // TODO: Select metadata extractions from `index.config`
    // TODO: Support custom metadata extractions
    const pipeline = createIndexIngestionPipeline(
      reader,
      parser, // Deprecate all rag.splitter.
      [
        new TitleExtractor({ llm }),
        new SummaryExtractor({ llm }),
        new QuestionsAnsweredExtractor({ llm }),
        new KeywordExtractor({ llm }),
      ],
      embedding,
    );

    // Find if it was a previously indexed document node.
    const previousIndexNode = await getDb().selectFrom('llamaindex_document_node')
      .select(eb => eb.fn('bin_to_uuid', ['id']).as('id'))
      .where('document_id', '=', document.id)
      .where('index_id', '=', index.id)
      .executeTakeFirst();

    // Major index procedure
    const result = await pipeline(document, previousIndexNode?.id);

    // Fill indexing task info to `document_index_task`
    // TODO: count tokens each LLM used
    mutableInfo.chunks_count = result.chunks.length;

    // Store index result into database
    const id = await tx(async () => {
      let allRelationships: Record<UUID, Record<NodeRelationship, RelatedNodeType<any>>> = {};

      const { id, relationships, ...metadata } = result.metadata;

      allRelationships[id] = relationships;

      await getDb().insertInto('llamaindex_document_node')
        .values({
          id: uuidToBin(id),
          metadata: JSON.stringify(metadata),
          hash: result.hash,
          text: result.content.join('\n\n'),
          document_id: document.id,
          index_id: index.id,
          indexed_at: new Date(),
          index_info: JSON.stringify({ index_config: index.config }),
        })
        .onDuplicateKeyUpdate({
          metadata: JSON.stringify(metadata),
          hash: result.hash,
          text: result.content.join('\n\n'),
          indexed_at: new Date(),
          index_info: JSON.stringify({ index_config: index.config }),
        })
        .execute();

      await getDb().insertInto(`llamaindex_document_chunk_node_${index.name}`)
        .values(result.chunks.map(chunk => {
          const { id, relationships, ...metadata } = chunk.metadata;
          allRelationships[id] = relationships;
          return {
            metadata: JSON.stringify(metadata),
            embedding: vectorToSql(chunk.vector),
            index_id: index.id,
            id: uuidToBin(id),
            document_id: document.id,
            text: chunk.content,
            hash: chunk.hash,
          };
        }))
        .execute();

      await getDb().insertInto('llamaindex_node_relationship')
        .values(Object.entries(allRelationships).reduce((arr: InsertObject<DBv1, 'llamaindex_node_relationship'>[], [sourceId, value]) => {
          Object.entries(value).forEach(([rel, itemOrArray]) => {
            if (itemOrArray instanceof Array) {
              itemOrArray.forEach(item => {
                arr.push({
                  source_node_id: uuidToBin(sourceId as UUID),
                  target_node_id: uuidToBin(item.nodeId as UUID),
                  type: rel as any,
                });
              });
            } else {
              arr.push({
                source_node_id: uuidToBin(sourceId as UUID),
                target_node_id: uuidToBin(itemOrArray.nodeId as UUID),
                type: rel as any,
              });
            }
          });
          return arr;
        }, []))
        .execute();

      return id;
    });

    return {
      documentNode: id,
      documentChunkNodeTableName: `llamaindex_document_chunk_node_${index.name}`,
      documentNodeTableName: `llamaindex_document_node`,
      isNewDocumentNode: !previousIndexNode,
    };
  };
}
