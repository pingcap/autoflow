import * as d3 from 'd3';

import type { IdType, NetworkLink, NetworkNode, ReadonlyNetwork } from './Network';
import type { SimulationLinkDatum, SimulationNodeDatum } from 'd3';

import ForceGraph from 'force-graph';
import type { NetworkRendererOptions } from './NetworkRendererOptions';

export interface NetworkNodeView extends SimulationNodeDatum {
  id: IdType;
  index: number;
  radius: number;
  label?: string;
  details?: string;
  meta?: any;
}

export interface NetworkLinkView extends SimulationLinkDatum<NetworkNodeView> {
  id: IdType;
  index: number;
  source: NetworkNodeView;
  target: NetworkNodeView;
  label?: string;
  details?: string;
  meta?: any;
}

export class CanvasNetworkRenderer<Node extends NetworkNode, Link extends NetworkLink> {
  private _el: HTMLElement | undefined;
  private _graph: any; // ForceGraph instance
  private _ro: ResizeObserver | undefined;

  private _onUpdateLink: ((id: IdType) => void) | undefined;
  private _onUpdateNode: ((id: IdType) => void) | undefined;

  private nodes: NetworkNodeView[] = [];
  private links: NetworkLinkView[] = [];

  // Graph state
  private selectedNode: NetworkNodeView | null = null;
  private selectedLink: NetworkLinkView | null = null;
  private highlightedNodes = new Set<IdType>();
  private highlightedLinks = new Set<IdType>();

  private readonly linkDefaultDistance = 30;
  private readonly chargeDefaultStrength = -80;
  private readonly linkHighlightDistance = 120;
  private readonly chargeHighlightStrength = -200;
  private readonly linkDefaultWidth = 1;

  private clustersCalculated = false;
  
  private adjacencyMap = new Map<IdType, { connectedNodes: Set<IdType>, connectedLinks: Set<IdType> }>();
  private adjacencyCalculated = false;

  scale = 1;
  private initialLayoutComplete = false;

  private viewportBounds = { x0: -Infinity, y0: -Infinity, x1: Infinity, y1: Infinity };

  private colors = {
    textColor: '#000000',
    nodeHighlighted: '#18a0b1',
    nodeSelected: '#72fefb',
    linkDefaultColor: '#999999',
    linkHighlighted: '#18a0b1',
    linkSelected: '#72fefb'
  };
  private zoomLevels = {
    one: 0.1,
    two: 0.2,
    three: 0.3,
    four: 0.4,
    five: 0.8,
  }

  constructor(
    private network: ReadonlyNetwork<Node, Link>,
    private options: NetworkRendererOptions<Node, Link>,
  ) {
    this.compile(options);
  }

  private compile(options: NetworkRendererOptions<Node, Link>) {
    const nodeMap = new Map<IdType, number>();
    this.nodes = this.network.nodes().map((node, index) => {
      const nodeRadius = 8;
      const fontSize = Math.max(8, nodeRadius * 0.3);
      const label = options.getNodeLabel?.(node) ?? (node as any).name ?? node.id;
      const labelColor = options.getNodeLabelColor?.(node) ?? this.colors.textColor;
      
      nodeMap.set(node.id, index);
      return {
        id: node.id,
        index,
        radius: nodeRadius, 
        label,
        details: options.getNodeDetails?.(node),
        meta: options.getNodeMeta?.(node),
        fontSize,
        fontString: `${fontSize}px Sans-Serif`,
        labelColor,
        ...options.getNodeInitialAttrs?.(node, index),
      };
    });
    this.links = this.network.links().map((link, index) => ({
      id: link.id,
      index,
      source: this.nodes[nodeMap.get(link.source)!],
      target: this.nodes[nodeMap.get(link.target)!],
      label: options.getLinkLabel?.(link),
      details: options.getLinkDetails?.(link),
      meta: options.getLinkMeta?.(link),
    }));
  }

  private updateViewportBounds() {
    if (!this._graph || !this._el) return;
    
    const canvas = this._el.querySelector('canvas');
    if (!canvas) return;

    const width = canvas.width;
    const height = canvas.height;
    
    const topLeft = this._graph.screen2GraphCoords(0, 0);
    const bottomRight = this._graph.screen2GraphCoords(width, height);
    
    const padding = 100 / this.scale;
    
    this.viewportBounds = {
      x0: topLeft.x - padding,
      y0: topLeft.y - padding,
      x1: bottomRight.x + padding,
      y1: bottomRight.y + padding
    };
  }

  private isNodeInViewport(node: any): boolean {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    return x >= this.viewportBounds.x0 && 
           x <= this.viewportBounds.x1 && 
           y >= this.viewportBounds.y0 && 
           y <= this.viewportBounds.y1;
  }

  private isLinkInViewport(link: any): boolean {
    const sourceX = link.source.x ?? 0;
    const sourceY = link.source.y ?? 0;
    const targetX = link.target.x ?? 0;
    const targetY = link.target.y ?? 0;
    
    if ((sourceX < this.viewportBounds.x0 && targetX < this.viewportBounds.x0) ||
        (sourceX > this.viewportBounds.x1 && targetX > this.viewportBounds.x1) ||
        (sourceY < this.viewportBounds.y0 && targetY < this.viewportBounds.y0) ||
        (sourceY > this.viewportBounds.y1 && targetY > this.viewportBounds.y1)) {
      return false;
    }
    
    return true;
  }

  mount(container: HTMLElement) {
    if (this._el) {
      return;
    }
    this._el = container;

    const { width: initialWidth, height: initialHeight } = container.getBoundingClientRect();

    const graph = new ForceGraph(container)
      .width(initialWidth)
      .height(initialHeight)
      .backgroundColor('transparent')
      .autoPauseRedraw(false)
      .warmupTicks(50)
      .nodeAutoColorBy('clusterId')
      .nodeCanvasObject((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        if (this.isNodeInViewport(node)) {
          this.drawNodeWithLabel(node, ctx, globalScale);
        }
      })
      .linkCanvasObject((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        this.scale = globalScale;
        if (this.scale > this.zoomLevels.three && this.isLinkInViewport(link)) {
          this.drawLink(link, ctx);
        }
      })
      .linkCanvasObjectMode(() => 'replace')
      .onNodeClick((node: any, event: MouseEvent) => {
        this.onNodeClick(node, event);
      })
      .onLinkClick((link: any, event: MouseEvent) => {
        this.onLinkClick(link, event);
      })
      .onBackgroundClick(() => {
        this.onBackgroundClick();
      })
      .d3Force('x', d3.forceX(0).strength(0.05))
      .d3Force('y', d3.forceY(0).strength(0.05))
      .d3Force("link", d3.forceLink().id((d: any) => d.id).distance(this.linkDefaultDistance))
      .d3Force("charge", d3.forceManyBody()
        .strength(this.chargeDefaultStrength)
        .theta(1.2)
      )
      .onZoom((transform: any) => {
        this.scale = transform.k;
      })
      .onRenderFramePre(() => {
        this.updateViewportBounds();
      });

    this._graph = graph;

    setTimeout(() => {
      this.initialLayoutComplete = true;
      graph.d3Force('x', null);
      graph.d3Force('y', null);
      graph.d3Force("charge")?.distanceMax(300).strength(0);

      const data = graph.graphData();
      data.nodes.forEach((node: any) => {
        node.fx = node.x;
        node.fy = node.y;
      });
    }, 2000);

    container.style.overflow = 'hidden';

    this._ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      this._graph.width(width).height(height);
    });
    this._ro.observe(container);

    this.render();

    setTimeout(() => {
      const canvas = container.querySelector('canvas');
      if (canvas) {
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.display = 'block';
      }
    }, 0);
  }

  unmount() {
    if (this._onUpdateLink) {
      this.network.off('update:link', this._onUpdateLink);
      this._onUpdateLink = undefined;
    }
    if (this._onUpdateNode) {
      this.network.off('update:node', this._onUpdateNode);
      this._onUpdateNode = undefined;
    }
    if (!this._el) {
      return;
    }

    if (this._graph) {
      this._graph.onNodeClick(null);
      this._graph.onLinkClick(null);
      this._graph.onBackgroundClick(null);
      
      this._graph.graphData({ nodes: [], links: [] });
      
      if (this._el) {
        this._el.innerHTML = '';
      }
      
      this._graph = undefined;
    }
    
    this._ro?.disconnect();
    this._ro = undefined;
    this._el = undefined;
  }

  private drawNodeWithLabel(node: any, ctx: CanvasRenderingContext2D, globalScale: number) {
    const nodeRadius = 8;
    const largeNodeRadius = 16;
    
    // Use different rendering based on zoom level
    if (globalScale < this.zoomLevels.one) {
      ctx.fillStyle = node.color;
      ctx.fillRect(node.x - nodeRadius/2, node.y - nodeRadius/2, largeNodeRadius, largeNodeRadius);
      return;
    }

    // Full circle rendering
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI, false);
    ctx.fillStyle = node.color;
    ctx.fill();

    // Selection/highlight strokes
    if (this.selectedNode && this.selectedNode.id === node.id) {
      ctx.strokeStyle = this.colors.nodeSelected;
      ctx.lineWidth = 3;
      ctx.stroke();
    } else if (this.highlightedNodes.has(node.id)) {
      ctx.strokeStyle = this.colors.nodeHighlighted;
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Labels only when zoomed in enough
    if (globalScale >= this.zoomLevels.five) {
      ctx.font = node.fontString;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = node.labelColor;
      ctx.fillText(node.label, node.x, node.y + nodeRadius + node.fontSize * 0.7);
    }
  }

  private drawLink(link: any, ctx: CanvasRenderingContext2D) {
    const source = link.source;
    const target = link.target;
    
    // Determine link color
    let color = this.colors.linkDefaultColor;
    if (this.selectedLink && this.selectedLink.id === link.id) {
      color = this.colors.linkSelected;
    } else if (this.highlightedLinks.has(link.id)) {
      color = this.colors.linkHighlighted;
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.min(this.linkDefaultWidth * this.scale, 1);
    
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.stroke();
    
    if (this.scale > this.zoomLevels.four) {
      this.drawArrow(ctx, source, target, color);
    }
  }

  private drawArrow(
    ctx: CanvasRenderingContext2D,
    source: any,
    target: any,
    color: string
  ) {
    const arrowLength = 6;
    const arrowAngle = Math.PI / 6;
    
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const angle = Math.atan2(dy, dx);
    
    const targetRadius = target.radius || 8;
    const arrowX = target.x - Math.cos(angle) * targetRadius;
    const arrowY = target.y - Math.sin(angle) * targetRadius;
    
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - arrowLength * Math.cos(angle - arrowAngle),
      arrowY - arrowLength * Math.sin(angle - arrowAngle)
    );
    ctx.lineTo(
      arrowX - arrowLength * Math.cos(angle + arrowAngle),
      arrowY - arrowLength * Math.sin(angle + arrowAngle)
    );
    ctx.closePath();
    ctx.fill();
  }

  private onNodeClick(node: any, event: MouseEvent) {
    this.selectedNode = node;
    this.selectedLink = null;
    
    this.options.onClickNode?.(this.network.node(node.id)!, event);
    this.highlightConnections(node);
  }

  private onLinkClick(link: any, event: MouseEvent) {
    this.selectedLink = link;
    this.selectedNode = null;
    
    this.options.onClickLink?.(this.network.link(link.id)!, event);
    this.highlightLink(link);
  }

  private onBackgroundClick() {
    this.options.onClickCanvas?.();
    this.clearHighlight();
  }

  private highlightLink(link: any) {
    this.highlightedLinks.clear();
    this.highlightedLinks.add(link.id);
  }

  private highlightConnections(node: any) {
    const adjacency = this.adjacencyMap.get(node.id);
    if (!adjacency) {
      return;
    }
    
    const connectedNodeIds = adjacency.connectedNodes;
    const connectedLinkIds = adjacency.connectedLinks;
    
    this.highlightedNodes.clear();
    connectedNodeIds.forEach(nodeId => this.highlightedNodes.add(nodeId));
    
    this.highlightedLinks.clear();
    connectedLinkIds.forEach(linkId => this.highlightedLinks.add(linkId));

    const data = this._graph.graphData();
    data.nodes.forEach((n: any) => {
      if (connectedNodeIds.has(n.id)) {
        n.fx = null;
        n.fy = null;
      }
    });
    
    this._graph.d3Force("link").distance((link: any) => {
      if (connectedLinkIds.has(link.id)) {
        return this.linkHighlightDistance;
      }
      return this.linkDefaultDistance;
    });

    this._graph.d3Force("charge").strength((node: any) => {
      if (connectedNodeIds.has(node.id)) {
        return this.chargeHighlightStrength;
      }
      return 0;
    });

    this._graph.d3ReheatSimulation();
  }

  private clearHighlight() {
    this.selectedNode = null;
    this.selectedLink = null;
    this.highlightedNodes.clear();
    this.highlightedLinks.clear();
    
    this._graph.d3Force("link").distance(this.linkDefaultDistance);
    if (!this.initialLayoutComplete) return;
    this._graph.d3Force("charge").strength(0);
    
    setTimeout(() => {
      const data = this._graph.graphData();
      data.nodes.forEach((n: any) => {
        n.fx = n.x;
        n.fy = n.y;
      });
    }, 500);
    
    this._graph.d3ReheatSimulation();
  }

  private calculateAndCacheClusters() {
    if (!this.nodes || !this.links) {
      return;
    }
    const clusters = this.findClusters();
    
    this.nodes.forEach(node => {
      const clusterId = clusters.get(node.id) || 0;
      (node as any).clusterId = clusterId;
    });
    
    this.clustersCalculated = true;
  }
  
  private calculateAndCacheAdjacency() {
    if (!this.nodes || !this.links) {
      return;
    }
    
    this.adjacencyMap.clear();
    
    this.nodes.forEach(node => {
      this.adjacencyMap.set(node.id, {
        connectedNodes: new Set<IdType>(),
        connectedLinks: new Set<IdType>()
      });
    });
    
    this.links.forEach(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      
      const sourceAdjacency = this.adjacencyMap.get(sourceId);
      const targetAdjacency = this.adjacencyMap.get(targetId);
      
      if (sourceAdjacency && targetAdjacency) {
        sourceAdjacency.connectedNodes.add(targetId);
        targetAdjacency.connectedNodes.add(sourceId);
        
        sourceAdjacency.connectedLinks.add(link.id);
        targetAdjacency.connectedLinks.add(link.id);
      }
    });
    
    this.adjacencyCalculated = true;
  }

  private findClusters(): Map<IdType, number> {
    const clusters = new Map<IdType, number>();
    const visited = new Set<IdType>();
    let clusterId = 0;
    
    const adjacencyList = new Map<IdType, IdType[]>();
    this.nodes.forEach(node => {
      adjacencyList.set(node.id, []);
    });
    
    this.links.forEach(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      
      if (adjacencyList.has(sourceId) && adjacencyList.has(targetId)) {
        adjacencyList.get(sourceId)!.push(targetId);
        adjacencyList.get(targetId)!.push(sourceId);
      }
    });
    
    const dfs = (nodeId: IdType, currentClusterId: number) => {
      if (visited.has(nodeId)) return;
      
      visited.add(nodeId);
      clusters.set(nodeId, currentClusterId);
      
      const neighbors = adjacencyList.get(nodeId) || [];
      for (const neighborId of neighbors) {
        if (!visited.has(neighborId)) {
          dfs(neighborId, currentClusterId);
        }
      }
    };

    this.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        dfs(node.id, clusterId);
        clusterId++;
      }
    });
    
    return clusters;
  }

  render() {
    if (!this.clustersCalculated) {
      this.calculateAndCacheClusters();
    }
    
    if (!this.adjacencyCalculated) {
      this.calculateAndCacheAdjacency();
    }

    const graphData = {
      nodes: this.nodes,
      links: this.links
    };
    
    this._graph.graphData(graphData);

    this._onUpdateNode = (id: IdType) => {
      const nodeIndex = this.nodes.findIndex(n => n.id === id);
      if (nodeIndex !== -1) {
        const networkNode = this.network.node(id);
        if (networkNode) {
          this.nodes[nodeIndex].label = this.options.getNodeLabel?.(networkNode);
          this.nodes[nodeIndex].details = this.options.getNodeDetails?.(networkNode);
          this.nodes[nodeIndex].meta = this.options.getNodeMeta?.(networkNode);
          
          this._graph.graphData({
            nodes: this.nodes,
            links: this.links
          });
        }
      }
    };

    this._onUpdateLink = (id: IdType) => {
      const linkIndex = this.links.findIndex(l => l.id === id);
      if (linkIndex !== -1) {
        const networkLink = this.network.link(id);
        if (networkLink) {
          this.links[linkIndex].label = this.options.getLinkLabel?.(networkLink);
          this.links[linkIndex].details = this.options.getLinkDetails?.(networkLink);
          this.links[linkIndex].meta = this.options.getLinkMeta?.(networkLink);
          
          this._graph.graphData({
            nodes: this.nodes,
            links: this.links
          });
        }
      }
    };

    this.network.on('update:node', this._onUpdateNode);
    this.network.on('update:link', this._onUpdateLink);

    setTimeout(() => {
      if (this._graph && this._graph.zoomToFit) {
        this._graph.zoomToFit(400, 50);
      }
    }, 1000);
  }

  // maintain for compatibility with NetworkRenderer
  focusNode(id: IdType): void {}
  blurNode(): void {}
  focusLink(id: IdType): void {}
  blurLink(): void {}
}