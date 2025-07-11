import { authenticationHeaders, handleResponse, requestUrl } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { bufferedReadableStreamTransformer } from '@/lib/buffered-readable-stream';
import { z, type ZodType } from 'zod';

export interface KnowledgeGraph {
  entities: KnowledgeGraphEntity[];
  relationships: KnowledgeGraphRelationship[];
}

export const enum KnowledgeGraphEntityType {
  original = 'original',
  synopsis = 'synopsis',
}

export interface KnowledgeGraphEntity {
  id: number;
  knowledge_base_id?: number | null;
  name: string;
  description: string;
  meta: object;
  entity_type: KnowledgeGraphEntityType;
  synopsis_info?: {
    entities: number[]
    topic: string
  } | null;
}

export interface KnowledgeGraphRelationship {
  id: number;
  source_entity_id: number;
  target_entity_id: number;
  knowledge_base_id?: number | null;
  description: string;
  meta: object;
  weight: number;
}

export const entitySchema = z.object({
  id: z.number(),
  knowledge_base_id: z.number().nullable().optional(),
  name: z.string(),
  description: z.string(),
  meta: z.object({}).passthrough(),
  entity_type: z.enum([KnowledgeGraphEntityType.original, KnowledgeGraphEntityType.synopsis]),
  synopsis_info: z.object({
    entities: z.number().array(),
    topic: z.string(),
  }).nullish(),
}) satisfies ZodType<KnowledgeGraphEntity>;

export const relationshipSchema = z.object({
  id: z.number(),
  knowledge_base_id: z.number().nullable().optional(),
  source_entity_id: z.number(),
  target_entity_id: z.number(),
  description: z.string(),
  last_modified_at: zodJsonDate().nullish(),
  meta: z.object({}).passthrough(),
  weight: z.number(),
}) satisfies ZodType<KnowledgeGraphRelationship>;

export const knowledgeGraphSchema = z.object({
  entities: entitySchema.array(),
  relationships: relationshipSchema.array(),
}) satisfies ZodType<KnowledgeGraph>;

export interface UpdateEntityParams {
  name: string | null;
  description: string | null;
  meta: object | null;
}

export interface CreateSynopsisEntityParams {
  name: string;
  description: string;
  meta: object;
  topic: string;
  entities: number[];
}

export interface UpdateRelationshipParams {
  description: string | null;
  meta: object | null;
  weight: number | null;
}

export interface GraphSearchParams {
  query: string;
  include_meta?: boolean;
  depth?: number;
  with_degree?: boolean;
}

export interface KBRetrieveKnowledgeGraphParams {
  query: string;
  llm_id: number;
  retrieval_config: {
    knowledge_graph: KnowledgeGraphRetrievalConfig;
  }
}

export interface KnowledgeGraphRetrievalConfig {
  depth?: number;
  include_meta?: boolean;
  with_degree?: boolean;
  metadata_filter?: {
    enabled?: boolean;
    filters?: Record<string, any>
  }
}

export async function search (kbId: number, params: GraphSearchParams) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/search`), {
    method: 'post',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  }).then(handleResponse(knowledgeGraphSchema));
}

export async function searchEntity (kbId: number, query: string, top_k: number = 10) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/entities/search`, { query, top_k }), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(entitySchema.array()));
}

export async function getEntity (kbId: number, id: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/entities/${id}`), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(entitySchema));
}

export async function updateEntity (kbId: number, id: number, params: UpdateEntityParams) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/entities/${id}`), {
    method: 'put',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  }).then(handleResponse(entitySchema));
}

export async function createSynopsisEntity (kbId: number, params: CreateSynopsisEntityParams) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/entities/synopsis`), {
    method: 'post',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  }).then(handleResponse(entitySchema));
}

export async function getEntitySubgraph (kbId: number, id: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/entities/${id}/subgraph`), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(knowledgeGraphSchema));
}

export async function getEntireKnowledgeGraph (kbId: number, params: KBRetrieveKnowledgeGraphParams) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/entire_graph`), {
    method: 'post',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
  .then(handleResponse(knowledgeGraphSchema));
}

export async function streamEntireKnowledgeGraph (kbId: number): Promise<KnowledgeGraph> {
  const entities: KnowledgeGraphEntity[] = [];
  const relationships: KnowledgeGraphRelationship[] = [];
  
  const response = await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/entire_graph/stream`), {
    method: 'GET',
    headers: await authenticationHeaders(),
    credentials: 'include',
  });
  
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  
  if (!response.body) {
    throw new Error('Empty response body');
  }
  
  const reader = response.body.pipeThrough(bufferedReadableStreamTransformer()).getReader();
  
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      if (value.trim() && value.startsWith('data: ')) {
        const dataStr = value.substring(6).trim();
        if (dataStr) {
          try {
            const data = JSON.parse(dataStr);
            
            if (data.type === 'entities') {
              entities.push(...data.data);
              console.log(`Received ${data.data.length} entities, total: ${entities.length}`);
            } else if (data.type === 'relationships') {
              relationships.push(...data.data);
              console.log(`Received ${data.data.length} relationships, total: ${relationships.length}`);
            } else if (data.type === 'complete') {
              console.log(`Streaming complete. Final counts - entities: ${entities.length}, relationships: ${relationships.length}`);
              return { entities, relationships };
            }
          } catch (error) {
            console.warn('Failed to parse streaming data:', error, 'Data:', dataStr);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  
  return { entities, relationships };
}

export async function getRelationship (kbId: number, id: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/relationships/${id}`), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(relationshipSchema));
}

export async function updateRelationship (kbId: number, id: number, params: UpdateRelationshipParams) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kbId}/graph/relationships/${id}`), {
    method: 'put',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  }).then(handleResponse(relationshipSchema));
}
