import type { IdType, NetworkLink, NetworkNode, ReadonlyNetwork } from '../network/Network';
import { useEffect, useRef, useState } from 'react';

import { CanvasNetworkRenderer } from '../network/CanvasNetworkRenderer';
import type { NetworkRendererOptions } from '../network/NetworkRendererOptions';

export interface NetworkCanvasProps<Node extends NetworkNode, Link extends NetworkLink> extends NetworkRendererOptions<Node, Link> {
  network: ReadonlyNetwork<Node, Link>;
  target: { type: string, id: IdType } | undefined;
  className?: string;
}

export function NetworkCanvas<Node extends NetworkNode, Link extends NetworkLink> ({ className, network, target, ...options }: NetworkCanvasProps<Node, Link>) {
  const ref = useRef<HTMLDivElement>(null);
  const [renderer, setRenderer] = useState<CanvasNetworkRenderer<Node, Link>>();

  useEffect(() => {
    // Cleanup previous renderer if it exists
    if (renderer) {
      renderer.unmount();
    }

    const newRenderer = new CanvasNetworkRenderer(network, options);
    
    if (ref.current) {
      newRenderer.mount(ref.current);
    }
    setRenderer(newRenderer);

    return () => {
      newRenderer.unmount();
      setRenderer(undefined);
    };
  }, [network]);

  useEffect(() => {
    if (!renderer) {
      return;
    }
    if (!target) {
      renderer.blurNode();
      renderer.blurLink();
      return;
    }
    switch (target.type) {
      case 'node':
        renderer.focusNode(target.id);
        break;
      case 'link':
        renderer.focusLink(target.id);
        break;
    }
  }, [target, renderer]);

  return (
    <div className={className} ref={ref} />
  );
}