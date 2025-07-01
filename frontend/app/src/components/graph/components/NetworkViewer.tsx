import { Loader } from '@/components/loader';
import { cn } from '@/lib/utils';
import { type FC, type ReactNode, useMemo, useState } from 'react';
import { NetworkCanvas } from '../components/NetworkCanvas';
import { NetworkContext } from '../components/NetworkContext';
import { BaseNetwork, type IdType } from '../network/Network';
import type { NetworkRendererOptions } from '../network/NetworkRenderer';
import { type Entity, type Relationship } from '../utils';

export interface NetworkViewerProps {
  className?: string;
  network: BaseNetwork<Entity, Relationship>;
  loading: boolean;
  loadingTitle: ReactNode;
  Details: FC<NetworkViewerDetailsProps>;
  useCanvasRenderer?: boolean;
}

export interface NetworkViewerDetailsProps {
  network: BaseNetwork<Entity, Relationship>,
  target: { type: string, id: IdType } | undefined,
  onTargetChange: ((target: { type: string, id: IdType } | undefined) => void)
}

function randomPosition (radius: number, kbSpacing: number, kbIndex: number, kbCount: number) {
  const x = kbIndex * kbSpacing - (kbCount - 1) * kbSpacing / 2;
  const theta = Math.random() * 2 * Math.PI;

  return {
    x: x + radius * Math.cos(theta),
    y: radius * Math.sin(theta),
  };
}

export function NetworkViewer ({ network, loading, loadingTitle, className, Details, useCanvasRenderer = false }: NetworkViewerProps) {
  const [target, setTarget] = useState<{ type: string, id: IdType }>();

  const knowledgeGraphIndexMap = useMemo(() => {
    const nodes = network.nodes();
    const kbIds = Array.from(nodes.reduce((acc, node) => acc.add(node.knowledge_base_id ?? 0), new Set<number>()));
    kbIds.sort();

    return new Map(kbIds.map((kbId, index) => ([kbId, index])));
  }, [network]);

  const networkOptions: NetworkRendererOptions<Entity, Relationship> = {
    showId: true,
    getNodeInitialAttrs: (node) => {
      const kbIndex = knowledgeGraphIndexMap.get(node.knowledge_base_id ?? 0) ?? 0;
      return randomPosition(20, 100, kbIndex, knowledgeGraphIndexMap.size || 1);
    },
    getNodeLabel: node => node.name,
    getNodeDetails: node => node.description,
    getNodeRadius: node => Math.pow(Math.log(1 + (network.nodeNeighborhoods(node.id)?.size ?? 0)) / Math.log(2), 2) * 2 + 5,
    getNodeColor: node => {
      if (node.entity_type === 'synopsis') {
        return `hsl(var(--brand1-foreground))`;
      } else {
        const kbIndex = knowledgeGraphIndexMap.get(node.knowledge_base_id ?? 0);
        if (!kbIndex) {
          return `hsl(var(--primary))`;
        } else {
          return `hsl(var(--chart-${kbIndex + 1}))`;
        }
      }
    },
    getNodeStrokeColor: node => {
      if (node.entity_type === 'synopsis') {
        return `hsl(var(--brand1))`;
      } else {
        const kbIndex = knowledgeGraphIndexMap.get(node.knowledge_base_id ?? 0);
        if (!kbIndex) {
          return `hsl(var(--primary))`;
        } else {
          return `hsl(var(--chart-${kbIndex + 1}))`;
        }
      }
    },
    getNodeLabelColor: node => {
      if (node.entity_type === 'synopsis') {
        return `hsl(var(--brand1))`;
      } else {
        return `hsl(var(--primary))`;
      }
    },
    getNodeLabelStrokeColor: node => {
      if (node.entity_type === 'synopsis') {
        return `hsl(var(--brand1-foreground))`;
      } else {
        return `hsl(var(--primary-foreground))`;
      }
    },
    getNodeMeta: node => node.meta,
    getLinkColor: link => {
      if (link.meta.relationship_type === 'synopsis') {
        return `hsl(var(--brand1) / 50%)`;
      } else {
        const kbIndex = knowledgeGraphIndexMap.get(link.knowledge_base_id ?? 0);
        if (!kbIndex) {
          return `hsl(var(--primary) / 50%)`;
        } else {
          return `hsl(var(--chart-${kbIndex + 1}) / 50%)`;
        }
      }
    },
    getLinkLabel: link => {
      const source = network.node(link.source)!;
      const target = network.node(link.target)!;
      return link.description
        .replace(source.name + ' -> ', '')
        .replace(' -> ' + target.name, '');
    },
    getLinkDetails: link => link.description,
    getLinkMeta: link => link.meta,
    getLinkLabelColor: (link) => {
      if (link.meta.relationship_type === 'synopsis') {
        return `hsl(var(--brand1) / 50%)`;
      } else {
        const kbIndex = knowledgeGraphIndexMap.get(link.knowledge_base_id ?? 0);
        if (!kbIndex) {
          return `hsl(var(--primary) / 50%)`;
        } else {
          return `hsl(var(--chart-${kbIndex + 1}) / 50%)`;
        }
      }
    },
    getLinkLabelStrokeColor: () => {
      return `hsl(var(--primary-foreground) / 50%)`;
    },

    onClickNode: (node) => {
      setTarget({ type: 'node', id: node.id });
    },
    onClickLink: (link) => {
      setTarget({ type: 'link', id: link.id });
    },
    onClickCanvas: () => {
      setTarget(undefined);
    },
  };

  return (
    <NetworkContext.Provider value={network}>
      <div className={cn('relative', className)}>
        <NetworkCanvas
          className={cn('w-full h-full overflow-hidden')}
          network={network}
          target={target}
          useCanvasRenderer={useCanvasRenderer}
          {...networkOptions}
        />
        <Details
          network={network}
          target={target}
          onTargetChange={setTarget}
        />
        <Loader loading={loading}>
          {loadingTitle}
        </Loader>
      </div>
    </NetworkContext.Provider>
  );
}
