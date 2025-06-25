'use client';

import { getChatMessageSubgraph } from '@/api/chats';
import { getEntitySubgraph, getEntireKnowledgeGraph, type KnowledgeGraph, search } from '@/api/graph';
import { LinkDetails } from '@/components/graph/components/LinkDetails';
import { NetworkViewer, type NetworkViewerDetailsProps } from '@/components/graph/components/NetworkViewer';
import { NodeDetails } from '@/components/graph/components/NodeDetails';
import type { IdType } from '@/components/graph/network/Network';
import { useNetwork } from '@/components/graph/useNetwork';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { buttonVariants } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useSearchParam } from '@/components/use-search-param';
import { getErrorMessage } from '@/lib/errors';
import isHotkey from 'is-hotkey';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import useSWR from 'swr';

export function GraphEditor ({ knowledgeBaseId }: { knowledgeBaseId: number }) {
  const [query, setQuery] = useSearchParam('query', 'sample-question:What is TiDB?');

  const [key, fetcher] = getFetchInfo(knowledgeBaseId, query);

  const { data: span, isLoading, error } = useSWR(key, fetcher, { revalidateOnFocus: false });

  const network = useNetwork(span);

  const ref = useRef<HTMLDivElement>(null);

  return (
    <div className="p-4 space-y-4">
      <SubgraphSelector knowledgeBaseId={knowledgeBaseId} query={query} onQueryChange={setQuery} />
      {(error != null) && <Alert variant="destructive">
        <AlertTitle>Failed to fetch subgraph</AlertTitle>
        <AlertDescription>{getErrorMessage(error)}</AlertDescription>
      </Alert>}
      <div className="w-full flex gap-4">
        <div className="flex-1">
          <NetworkViewer
            key={query}
            className="border rounded h-auto aspect-square"
            loading={isLoading}
            loadingTitle={'Loading knowledge graph...'}
            network={network}
            Details={(props) => (
              ref.current && createPortal(
                <Editor
                  knowledgeBaseId={knowledgeBaseId}
                  {...props}
                  onEnterSubgraph={(type, id) => {
                    props.onTargetChange(undefined);
                    switch (type) {
                      case `entity`:
                        setQuery(`entity:${id}`);
                        break;
                      case 'document':
                        setQuery(`document:${id}`);
                        break;
                    }
                  }}
                />,
                ref.current!,
              )
            )}
          />
        </div>
        <div className="w-96 flex-shrink-0 relative" style={{ padding: '0.1px' }} ref={ref} />
      </div>
    </div>
  );
}

function SubgraphSelector ({ knowledgeBaseId, query, onQueryChange }: { knowledgeBaseId: number, query: string | null, onQueryChange: (query: string) => void }) {
  const [initialType = 'sample-question', initialInput = 'What is TiDB?'] = parseQuery(query) ?? [];

  const [type, setType] = useState<string>(initialType);
  const [input, setInput] = useState<string>(initialInput);

  useEffect(() => {
    const [type = 'sample-question', input = 'What is TiDB?'] = parseQuery(query) ?? [];
    setType(type);
    setInput(input);
  }, [query]);

  return (
    <div className="flex gap-2">
      <Select value={type} onValueChange={type => {
        setType(type);
        setInput('');
        if (type === 'entire-knowledge-graph') {
          onQueryChange(`${type}:`);
        }
      }}>
        <SelectTrigger className="w-max">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="sample-question">Sample Question</SelectItem>
          <SelectItem value="entity">Entity ID</SelectItem>
          <SelectItem value="message-subgraph">Message Subgraph</SelectItem>
          <SelectItem value="trace" disabled>Langfuse Trace ID (UUID)</SelectItem>
          <SelectItem value="document" disabled>Document URI</SelectItem>
          <SelectItem value="entire-knowledge-graph">Entire Knowledge Graph</SelectItem>
        </SelectContent>
      </Select>
      {type !== 'entire-knowledge-graph' && (
        <>
          <Input
            className="flex-1"
            value={input}
            onChange={event => setInput(event.target.value)}
            onKeyDown={event => {
              if (isHotkey('Enter', event)) {
                onQueryChange(`${type}:${input}`);
              }
            }}
          />
          <Link className={buttonVariants({})} href={`/knowledge-bases/${knowledgeBaseId}/knowledge-graph-explorer/create-synopsis-entity`}>
            Create Synopsis Entity
          </Link>
        </>
      )}
    </div>
  );
}

function Editor ({ knowledgeBaseId, network, target, onTargetChange, onEnterSubgraph }: NetworkViewerDetailsProps & { knowledgeBaseId: number, onEnterSubgraph: (type: string, entityId: IdType) => void }) {
  if (target) {
    if (target.type === 'link') {
      return <LinkDetails knowledgeBaseId={knowledgeBaseId} relationship={network.link(target.id)!} onClickTarget={onTargetChange} onEnterSubgraph={onEnterSubgraph} />;
    } else if (target.type === 'node') {
      return <NodeDetails knowledgeBaseId={knowledgeBaseId} entity={network.node(target.id)!} onClickTarget={onTargetChange} onEnterSubgraph={onEnterSubgraph} />;
    }
  }

  return <div className="flex items-center justify-center h-40 text-sm text-muted-foreground font-bold">
    Select an entity or relationship
  </div>;
}

function getFetchInfo (kbId: number, query: string | null): [string | false, () => Promise<KnowledgeGraph>] {
  if (!query) {
    return [false, () => Promise.reject()];
  }

  const parsedQuery = parseQuery(query);
  if (!parsedQuery) {
    return [false, () => Promise.reject()];
  }

  const param = parsedQuery[1];

  const entireKnowledgeGraphParams = {
    query: "",
    llm_id: 1,
    retrieval_config: {
      knowledge_graph: {
        depth: 20,
        include_meta: true,
        with_degree: true
      }
    }
  }

  switch (parsedQuery[0]) {
    // case 'trace':
    //   return ['get', `/api/v1/traces/${parsedQuery[1]}/knowledge-graph-retrieval`];
    // case 'document':
    //   return ['get', `/api/v1/indexes/${indexName}/chunks/${encodeURIComponent(parsedQuery[1])}/subgraph`];
    case 'entity':
      return [`api.knowledge-bases.${kbId}.graph.entity-subgraph?id=${param}`, () => getEntitySubgraph(kbId, parseInt(param))];
    case 'sample-question':
      return [`api.knowledge-bases.${kbId}.graph.search?query=${param}`, () => search(kbId, { query: param })];
    case 'message-subgraph':
      return [`api.chats.get-message-subgraph?id=${param}`, () => getChatMessageSubgraph(parseInt(param))];
    case 'entire-knowledge-graph':
      return [`api.knowledge-bases.${kbId}.graph.entire-knowledge-graph`, () => getEntireKnowledgeGraph(kbId, entireKnowledgeGraphParams)];
  }

  return [false, () => Promise.reject()];
}

function parseQuery (query: string | null) {
  if (!query) {
    return null;
  }

  const idx = query.indexOf(':');
  if (idx < 0) {
    return null;
  }

  return [query.slice(0, idx), query.slice(idx + 1)] as const;
}
