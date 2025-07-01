import { useEffect, useRef, useState } from 'react';
import type { IdType, NetworkLink, NetworkNode, ReadonlyNetwork } from '../network/Network';
import { NetworkRenderer, type NetworkRendererOptions } from '../network/NetworkRenderer';
import { CanvasNetworkRenderer } from '../network/CanvasNetworkRenderer';

export interface NetworkCanvasProps<Node extends NetworkNode, Link extends NetworkLink> extends NetworkRendererOptions<Node, Link> {
  network: ReadonlyNetwork<Node, Link>;
  target: { type: string, id: IdType } | undefined;
  className?: string;
  useCanvasRenderer?: boolean;
}

export function NetworkCanvas<Node extends NetworkNode, Link extends NetworkLink> ({ className, network, target, useCanvasRenderer = false, ...options }: NetworkCanvasProps<Node, Link>) {
  const ref = useRef<HTMLDivElement>(null);
  const [renderer, setRenderer] = useState<NetworkRenderer<Node, Link> | CanvasNetworkRenderer<Node, Link>>();

  useEffect(() => {
    // Cleanup previous renderer if it exists (needed for renderer switching)
    if (renderer) {
      renderer.unmount();
    }

    const newRenderer = useCanvasRenderer 
      ? new CanvasNetworkRenderer(network, options)
      : new NetworkRenderer(network, options);
    
    if (ref.current) {
      newRenderer.mount(ref.current);
    }
    setRenderer(newRenderer);

    return () => {
      newRenderer.unmount();
      setRenderer(undefined);
    };
  }, [network, useCanvasRenderer]);

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
        return () => renderer.blurNode();
      case 'link':
        renderer.focusLink(target.id);
        return () => renderer.blurLink();
    }
  }, [target, renderer]);

  return (
    <div className={className} ref={ref} />
  );
}